from __future__ import annotations

import argparse
import json
import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

from evaluation.metrics import evaluate_batch, workflow_completion_rate
from evaluation.schemas import BenchmarkItem, PredictionItem

from core.config import Config


def load_jsonl(path: Path, model):
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(model(**json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid JSONL row {line_number} in {path}: {exc}") from exc
    return items


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def aggregate(scores: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(scores)
    if not rows:
        return {}
    metric_names = sorted({name for row in rows for name in row})
    return {
        metric: mean(row.get(metric, 0.0) for row in rows)
        for metric in metric_names
    }


def trace_load_dataset(benchmark_path: Path, predictions_path: Path):
    benchmark = {item.id: item for item in load_jsonl(benchmark_path, BenchmarkItem)}
    predictions: List[PredictionItem] = load_jsonl(predictions_path, PredictionItem)
    return benchmark, predictions


def trace_predictions(predictions: List[PredictionItem]):
    return predictions


def save_evaluation_results(result: Dict[str, object], output_path: Path | None):
    if output_path is not None:
        write_json(output_path, result)


def evaluate_predictions(
    benchmark_path: Path, 
    predictions_path: Path, 
    output_path: Path | None = None
) -> Dict[str, object]:

    benchmark, predictions = trace_load_dataset(benchmark_path, predictions_path)

    # Validate predictions IDs
    valid_pairs = []
    for prediction in predictions:
        if prediction.id not in benchmark:
            raise ValueError(f"Prediction id {prediction.id!r} does not exist in benchmark.")
        valid_pairs.append((benchmark[prediction.id], prediction))

    print(f"\nRunning batch evaluation for {len(predictions)} items...")
    items = [pair[0] for pair in valid_pairs]
    preds = [pair[1] for pair in valid_pairs]
    
    # Trace the prediction presence
    trace_predictions(preds)
    
    batch_scores = evaluate_batch(items, preds)

    per_item = []
    result = {}
    for idx, (item, prediction) in enumerate(valid_pairs):
        item_scores = batch_scores[idx]
        per_item.append({
            "id": prediction.id,
            "query_type": item.query_type.value,
            "difficulty_level": item.difficulty_level.value,
            "scores": item_scores,
        })

        by_type: Dict[str, List[Dict[str, float]]] = {}
        by_difficulty: Dict[str, List[Dict[str, float]]] = {}
        for row in per_item:
            by_type.setdefault(row["query_type"], []).append(row["scores"])
            by_difficulty.setdefault(row["difficulty_level"], []).append(row["scores"])

        result = {
            "overall": {
                **aggregate(row["scores"] for row in per_item),
                "workflow_completion_rate": workflow_completion_rate(predictions[:len(per_item)]),
            },
            "by_query_type": {key: aggregate(value) for key, value in by_type.items()},
            "by_difficulty": {key: aggregate(value) for key, value in by_difficulty.items()},
            "per_item": per_item,
        }

        save_evaluation_results(result, output_path)

    if output_path is not None:
        print(f"Saved evaluation progress to {output_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DocuMind benchmark predictions.")
    parser.add_argument("--benchmark", type=Path, required=True, help="Benchmark JSONL file.")
    parser.add_argument("--predictions", type=Path, required=True, help="Predictions JSONL file.")
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/metrics.json"))
    args = parser.parse_args()

    result = evaluate_predictions(args.benchmark, args.predictions, output_path=args.output)
        
    print("\nFinal Aggregated Metrics:")
    print(json.dumps(result.get("overall", {}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
