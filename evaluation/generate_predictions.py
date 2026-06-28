import argparse
import json
import time
import datetime
from pathlib import Path
from typing import List

from core.orchestrator import Orchestrator
from vectorstore.chroma import VectorStoreManager
from evaluation.schemas import BenchmarkItem, PredictionItem, RetrievedContext
from evaluation.runner import load_jsonl

from core.config import Config


def parse_agent_state_to_prediction(item: BenchmarkItem, state, completed: bool) -> PredictionItem:
    # 1. Parse retrieved contexts
    retrieved_contexts = []
    for doc in state.get("retrieved_docs", []):
        meta = doc.get("metadata", {})
        retrieved_contexts.append(
            RetrievedContext(
                content=doc.get("content", ""),
                source_document=meta.get("source"),
                page_number=meta.get("page_number"),
                chunk_id=doc.get("chunk_id"),
                score=doc.get("score")
            )
        )
    
    # 2. Extract agent path from audit log steps
    agent_path = []
    for step in state.get("audit_log", []):
        if "step" in step and step["step"] not in agent_path:
            agent_path.append(step["step"])
            
    # 3. Extract tools selected
    selected_tools = []
    for msg in state.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                selected_tools.append(tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", str(tc)))
        elif isinstance(msg, dict) and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                selected_tools.append(tc.get("name") if isinstance(tc, dict) else str(tc))

    return PredictionItem(
        id=item.id,
        question=item.question,
        answer=state.get("final_response") or "",
        citations=state.get("citations") or [],
        retrieved_contexts=retrieved_contexts,
        route=state.get("route"),
        selected_tools=selected_tools,
        agent_path=agent_path,
        workflow_completed=completed
    )


def trace_load_dataset(benchmark_path: Path):
    return load_jsonl(benchmark_path, BenchmarkItem)


def save_prediction(pred_dict: dict, output_path: Path, update: bool, existing_preds: list, idx: int, force: bool):
    if update:
        # Find and replace the item in existing_preds, or append if not found
        replaced = False
        for i, p in enumerate(existing_preds):
            if p.get("id") == pred_dict.get("id"):
                existing_preds[i] = pred_dict
                replaced = True
                break
        if not replaced:
            existing_preds.append(pred_dict)
            
        # Write all predictions back to the file
        with output_path.open("w", encoding="utf-8") as out_file:
            for p in existing_preds:
                out_file.write(json.dumps(p) + "\n")
        print(f"Updated prediction ID: {pred_dict.get('id')} in {output_path}")
    else:
        # Overwrite on the first item if not updating and starting fresh, else append
        mode = "w" if (idx == 1 and (force or not existing_preds)) else "a"
        with output_path.open(mode, encoding="utf-8") as out_file:
            out_file.write(json.dumps(pred_dict) + "\n")
        print(f"Wrote prediction ID: {pred_dict.get('id')} to {output_path}")


def generate_all_predictions(
    benchmark_items: List[BenchmarkItem], 
    orchestrator: Orchestrator, 
    delay: float, 
    output_path: Path, 
    update: bool, 
    force: bool, 
    existing_preds: list
):
    print(f"Running predictions on {len(benchmark_items)} items...")
    for idx, item in enumerate(benchmark_items, start=1):
        if idx > 1 and delay > 0:
            print(f"Sleeping for {delay} seconds to avoid rate limits...")
            time.sleep(delay)
        print(f"[{idx}/{len(benchmark_items)}] Processing query ID: {item.id}...")
        try:
            # Execute SUT orchestrator graph
            result_state = orchestrator.run(item.question, query_id=item.id)
            completed = True
        except Exception as e:
            print(f"Error running query {item.id}: {e}")
            result_state = {
                "final_response": f"Error: {e}",
                "route": "error",
                "retrieved_docs": [],
                "citations": [],
                "audit_log": [{"step": "error"}],
                "messages": []
            }
            completed = False
        
        # Map state to PredictionItem
        prediction = parse_agent_state_to_prediction(item, result_state, completed)
        
        # Write to JSONL
        pred_dict = {
            "id": prediction.id,
            "question": prediction.question,
            "answer": prediction.answer,
            "citations": prediction.citations,
            "retrieved_contexts": [
                {
                    "content": ctx.content,
                    "source_document": ctx.source_document,
                    "page_number": ctx.page_number,
                    "citation": ctx.citation,
                    "chunk_id": ctx.chunk_id,
                    "score": ctx.score
                }
                for ctx in prediction.retrieved_contexts
            ],
            "route": prediction.route,
            "selected_tools": prediction.selected_tools,
            "agent_path": prediction.agent_path,
            "workflow_completed": prediction.workflow_completed,
            "metadata": prediction.metadata
        }
        
        save_prediction(pred_dict, output_path, update, existing_preds, idx, force)


def run_evaluation_predictions(
    benchmark_path: Path,
    output_path: Path,
    delay: float,
    query_id: str | None,
    update: bool,
    force: bool,
    vector_store: VectorStoreManager,
    orchestrator: Orchestrator
):

    # Load Benchmark Items via traceable Dataset function
    print(f"Loading benchmark from {benchmark_path}...")
    benchmark_items: List[BenchmarkItem] = trace_load_dataset(benchmark_path)
    
    if query_id:
        benchmark_items = [item for item in benchmark_items if item.id == query_id]
        if not benchmark_items:
            print(f"No item with ID {query_id} found in benchmark.")
            return
            
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing predictions if the output file exists
    existing_preds = []
    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        existing_preds.append(json.loads(line.strip()))
            print(f"Loaded {len(existing_preds)} existing predictions.")
        except Exception as e:
            print(f"Warning: Could not read existing predictions file: {e}")

    # Skip items already in existing predictions unless force is specified
    if not force and existing_preds:
        existing_ids = {p.get("id") for p in existing_preds if p.get("id")}
        original_count = len(benchmark_items)
        benchmark_items = [item for item in benchmark_items if item.id not in existing_ids]
        skipped_count = original_count - len(benchmark_items)
        if skipped_count > 0:
            print(f"Skipping {skipped_count} items already present in the predictions file.")

    if not benchmark_items:
        print("All benchmark items have already been processed. Nothing to run.")
        return

    generate_all_predictions(
        benchmark_items=benchmark_items,
        orchestrator=orchestrator,
        delay=delay,
        output_path=output_path,
        update=update,
        force=force,
        existing_preds=existing_preds
    )


def main():
    parser = argparse.ArgumentParser(description="Generate DocuMind predictions for a benchmark.")
    parser.add_argument("--benchmark", type=Path, default=Path("evaluation/benchmark.jsonl"), help="Path to the benchmark JSONL file.")
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/predictions.jsonl"), help="Path to save predictions JSONL file.")
    parser.add_argument("--delay", type=float, default=30.0, help="Delay in seconds between executing predictions to avoid rate limits.")
    parser.add_argument("--query-id", type=str, help="Run prediction only for this specific query ID.")
    parser.add_argument("--update", action="store_true", help="Update the entry in the existing output file instead of overwriting.")
    parser.add_argument("--force", action="store_true", help="Force re-running predictions even if they already exist in the output file.")
    args = parser.parse_args()
    
    # Initialize DocuMind components
    print("Initializing Vector Store and Orchestrator...")
    vector_store = VectorStoreManager()
    orchestrator = Orchestrator(vector_store=vector_store)
    
    run_evaluation_predictions(
        benchmark_path=args.benchmark,
        output_path=args.output,
        delay=args.delay,
        query_id=args.query_id,
        update=args.update,
        force=args.force,
        vector_store=vector_store,
        orchestrator=orchestrator
    )
    print(f"Done!")


if __name__ == "__main__":
    main()
