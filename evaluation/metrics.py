from __future__ import annotations


import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set

from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.llms.base import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)

from core.llm import get_llm
from core.config import Config
from evaluation.schemas import BenchmarkItem, PredictionItem, RetrievedContext


import asyncio
import threading
import time

CALLS_PER_MINUTE = 5
MIN_GAP = 60.0 / CALLS_PER_MINUTE  # 7.5s


class RateLimitedFallbackRagasLLM(LangchainLLMWrapper):
    def __init__(self, primary_llm, fallbacks: list, cache=None, run_config=None):
        super().__init__(primary_llm, cache=cache, run_config=run_config)
        # Accept a list of fallbacks for chained failover
        self.fallback_wrappers = [
            LangchainLLMWrapper(f, cache=cache, run_config=run_config)
            for f in fallbacks
        ]
        self._thread_lock = threading.Lock()
        self._last_call_time = 0.0
        self._loop_locks = {}

    def _get_async_lock(self):
        """Per-event-loop lock to avoid 'bound to different loop' error."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop_id = id(loop)
        with self._thread_lock:
            if loop_id not in self._loop_locks:
                self._loop_locks[loop_id] = asyncio.Lock()
        return self._loop_locks[loop_id]

    def _update_last_call(self):
        with self._thread_lock:
            self._last_call_time = time.time()

    def _get_sleep_time(self):
        with self._thread_lock:
            elapsed = time.time() - self._last_call_time
            return max(0.0, MIN_GAP - elapsed)

    def _try_fallbacks_sync(self, prompt, n, temperature, stop, callbacks, original_error):
        for i, wrapper in enumerate(self.fallback_wrappers):
            try:
                name = getattr(wrapper.langchain_llm, 'model_name', None) \
                    or getattr(wrapper.langchain_llm, 'model', f'fallback-{i}')
                print(f"[Ragas LLM] Trying fallback {i+1}: {name}")
                return wrapper.generate_text(prompt, n=1, temperature=temperature,
                                             stop=stop, callbacks=callbacks)
            except Exception as e:
                print(f"[Ragas LLM] Fallback {i+1} also failed: {e}")
        raise original_error

    async def _try_fallbacks_async(self, prompt, n, temperature, stop, callbacks, original_error):
        for i, wrapper in enumerate(self.fallback_wrappers):
            try:
                name = getattr(wrapper.langchain_llm, 'model_name', None) \
                    or getattr(wrapper.langchain_llm, 'model', f'fallback-{i}')
                print(f"[Ragas LLM] Trying async fallback {i+1}: {name}")
                return await wrapper.agenerate_text(prompt, n=1, temperature=temperature,
                                                    stop=stop, callbacks=callbacks)
            except Exception as e:
                print(f"[Ragas LLM] Async fallback {i+1} also failed: {e}")
        raise original_error

    def generate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
        requested_n = n
        to_sleep = self._get_sleep_time()
        if to_sleep > 0:
            time.sleep(to_sleep)
        try:
            res = super().generate_text(prompt, 1, temperature, stop, callbacks)
            if requested_n > 1 and res.generations:
                single_gen = res.generations[0][0]
                res.generations[0] = [single_gen] * requested_n
            return res
        except Exception as e:
            print(f"[Ragas LLM] Primary failed: {e}")
            res = self._try_fallbacks_sync(prompt, 1, temperature, stop, callbacks, e)
            if requested_n > 1 and res.generations:
                single_gen = res.generations[0][0]
                res.generations[0] = [single_gen] * requested_n
            return res
        finally:
            self._update_last_call()

    async def agenerate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
        requested_n = n
        lock = self._get_async_lock()  # per-loop lock — fixes RuntimeError
        async with lock:
            to_sleep = self._get_sleep_time()
            if to_sleep > 0:
                print(f"[RateLimiter] Async sleeping {to_sleep:.2f}s...")
                await asyncio.sleep(to_sleep)
            try:
                res = await super().agenerate_text(prompt, 1, temperature, stop, callbacks)
                if requested_n > 1 and res.generations:
                    single_gen = res.generations[0][0]
                    res.generations[0] = [single_gen] * requested_n
                return res
            except Exception as e:
                print(f"[Ragas LLM] Primary failed (async): {e}")
                res = await self._try_fallbacks_async(
                    prompt, 1, temperature, stop, callbacks, e
                )
                if requested_n > 1 and res.generations:
                    single_gen = res.generations[0][0]
                    res.generations[0] = [single_gen] * requested_n
                return res
            finally:
                self._update_last_call()


# 3-tier fallback chain: Groq 70B → Gemini Flash-Lite → Groq 8B
try:
    _primary  = get_llm("gemini/gemini-3.1-flash-lite",   temperature=0.0)
    _fb1      = get_llm("gemini/gemini-2.5-flash-lite",    temperature=0.0)  # different provider
    _fb2      = get_llm("gemini/gemini-2.1-flash",       temperature=0.0)  # separate TPD pool
    _evaluator_llm = RateLimitedFallbackRagasLLM(_primary, fallbacks=[_fb1, _fb2])
except Exception as e:
    print(f"Failed to initialize Ragas LLM judge: {e}")
    _evaluator_llm = None

# Initialize Embeddings model once
try:
    _embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    _evaluator_embeddings = LangchainEmbeddingsWrapper(_embeddings_model)
except Exception as e:
    print(f"Failed to initialize Ragas Embeddings adapter: {e}")
    _evaluator_embeddings = None


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower().strip())


def _tokens(value: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _source_key(source_document: Optional[str], page_number: Optional[int]) -> str:
    return f"{source_document or ''}::p{page_number or ''}".lower()


def gold_source_keys(item: BenchmarkItem) -> Set[str]:
    return {
        _source_key(src.source_document, src.page_number)
        for src in item.expected_sources
    }


def retrieved_source_keys(contexts: Sequence[RetrievedContext]) -> List[str]:
    return [
        _source_key(ctx.source_document, ctx.page_number)
        for ctx in contexts
    ]


def hit_rate_at_k(gold: Set[str], retrieved: Sequence[str], k: int) -> float:
    return float(any(key in gold for key in retrieved[:k]))


def mrr(gold: Set[str], retrieved: Sequence[str]) -> float:
    for index, key in enumerate(retrieved, start=1):
        if key in gold:
            return 1.0 / index
    return 0.0


def ndcg_at_k(relevance_by_key: Dict[str, int], retrieved: Sequence[str], k: int) -> float:
    def dcg(scores: Sequence[int]) -> float:
        return sum((2**score - 1) / math.log2(index + 2) for index, score in enumerate(scores))

    gains = [relevance_by_key.get(key, 0) for key in retrieved[:k]]
    ideal = sorted(relevance_by_key.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return dcg(gains) / ideal_dcg


def query_routing_accuracy(item: BenchmarkItem, prediction: PredictionItem) -> float:
    pred_route = (prediction.route or "").lower()
    expected_route = item.expected_route.lower()
    if expected_route == "document" and pred_route in ["document", "single_hop", "multi_hop"]:
        return 1.0
    return float(pred_route == expected_route)


def tool_selection_accuracy(item: BenchmarkItem, prediction: PredictionItem) -> float:
    expected = set(item.expected_tools)
    actual = set(prediction.selected_tools)
    if not expected and not actual:
        return 1.0
    if not expected:
        return 0.0
    return len(expected.intersection(actual)) / len(expected.union(actual))


def workflow_completion_rate(predictions: Iterable[PredictionItem]) -> float:
    predictions = list(predictions)
    if not predictions:
        return 0.0
    return sum(1 for prediction in predictions if prediction.workflow_completed) / len(predictions)



def compute_custom_metrics(items: Sequence[BenchmarkItem], predictions: Sequence[PredictionItem]) -> List[Dict[str, float]]:
    all_scores = []
    for item, prediction in zip(items, predictions):
        gold = gold_source_keys(item)
        retrieved = retrieved_source_keys(prediction.retrieved_contexts)
        relevance_by_key = {
            _source_key(src.source_document, src.page_number): src.relevance_grade
            for src in item.expected_sources
        }

        # Custom deterministic metrics
        scores: Dict[str, float] = {
            "hit_rate@3": hit_rate_at_k(gold, retrieved, 3),
            "ndcg@3": ndcg_at_k(relevance_by_key, retrieved, 3),
            "mrr": mrr(gold, retrieved),
            "query_routing_accuracy": query_routing_accuracy(item, prediction),
            "tool_selection_accuracy": tool_selection_accuracy(item, prediction),
        }
        all_scores.append(scores)
    return all_scores



def run_all_ragas_evaluations(questions, answers, contexts_lists, ground_truths, llm, embeddings):
    results_list = []
    for idx in range(len(questions)):
        print(f"\n[Ragas Evaluation] Evaluating item {idx + 1}/{len(questions)}...")
        dataset = Dataset.from_dict({
            "question": [questions[idx]],
            "answer": [answers[idx]],
            "contexts": [contexts_lists[idx]],
            "ground_truth": [ground_truths[idx]]
        })
        run_config = RunConfig(timeout=600, max_retries=10)
        try:
            res = evaluate(
                dataset=dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    # context_precision,  # Commented out to reduce API calls (NDCG@3 / MRR evaluated deterministically)
                    context_recall,
                    answer_correctness,
                ],
                llm=llm,
                embeddings=embeddings,
                run_config=run_config
            )
            results_list.append(res)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Ragas Evaluation] Error evaluating item {idx + 1}: {e}")
            results_list.append(None)
        
        # Sleep 60 seconds to cool down API rate limits before the next item
        if idx < len(questions) - 1:
            print("[Ragas Evaluation] Sleeping 60 seconds to cool down API rate limits...")
            time.sleep(60.0)
    return results_list


def evaluate_batch(items: Sequence[BenchmarkItem], predictions: Sequence[PredictionItem]) -> List[Dict[str, float]]:
    if not items:
        return []

    all_scores = compute_custom_metrics(items, predictions)

    questions = [item.question or "" for item in items]
    answers = [prediction.answer or "" for prediction in predictions]
    contexts_lists = [
        [ctx.content for ctx in prediction.retrieved_contexts] if prediction.retrieved_contexts else [""]
        for prediction in predictions
    ]
    ground_truths = [item.ground_truth_answer or "" for item in items]

    # Initialize default values
    for scores in all_scores:
        scores.update({
            "context_recall": 0.0,
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "answer_correctness": 0.0,
            "hallucination_rate": 1.0,
        })

    if _evaluator_llm is not None:
        try:
            results_list = run_all_ragas_evaluations(
                questions, answers, contexts_lists, ground_truths, 
                _evaluator_llm, _evaluator_embeddings
            )
            
            for idx, results in enumerate(results_list):
                if results is None:
                    print(f"[Ragas Evaluation] Item {idx + 1} evaluation was skipped or failed.")
                    continue
                df = results.to_pandas()
                if df.empty:
                    print(f"[Ragas Evaluation] Item {idx + 1} returned no results.")
                    continue
                row = df.iloc[0]
                def safe_float(val) -> float:
                    try:
                        fval = float(val)
                        return 0.0 if math.isnan(fval) or math.isinf(fval) else fval
                    except:
                        return 0.0

                ragas_faithfulness = safe_float(row.get("faithfulness", 0.0))
                ragas_relevancy = safe_float(row.get("answer_relevancy", 0.0))
                ragas_recall = safe_float(row.get("context_recall", 0.0))
                ragas_correctness = safe_float(row.get("answer_correctness", 0.0))

                print(f"[Ragas Evaluation] Item {idx + 1} Ragas scores:")
                print(f"  - Faithfulness:      {ragas_faithfulness:.4f}")
                print(f"  - Answer Relevancy:   {ragas_relevancy:.4f}")
                print(f"  - Context Recall:     {ragas_recall:.4f}")
                print(f"  - Answer Correctness: {ragas_correctness:.4f}")

                all_scores[idx].update({
                    "context_recall": ragas_recall,
                    "faithfulness": ragas_faithfulness,
                    "answer_relevancy": ragas_relevancy,
                    "answer_correctness": ragas_correctness,
                    "hallucination_rate": max(0.0, 1.0 - ragas_faithfulness),
                })
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Ragas evaluation failed: {e}")
    else:
        print("Ragas evaluator LLM is not initialized. Using 0.0 as default values.")

    return all_scores


def evaluate_pair(item: BenchmarkItem, prediction: PredictionItem) -> Dict[str, float]:
    return evaluate_batch([item], [prediction])[0]
