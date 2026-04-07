"""
evaluation.py — Ablation Study Evaluation
Owner: Nivi

Scores all configurations produced by ablation_runner.py using:

    Lexical metrics  (all queries, fast):
        • Exact Match (EM)
        • Token F1
        • Answer Success Rate

    Semantic metrics via RAGAS (sampled subset, slow):
        • Faithfulness
        • Answer Relevance
        • Context Recall

Sampling rationale
──────────────────
ablation_runner.py now produces 12 configs (4 ablation × 3 budgets).
At 150 queries each that is 1 800 RAGAS calls through phi3:mini — multiple
hours of serial LLM inference.  A stratified 20 % random sample (default)
reduces this to ~360 calls while preserving per-config coverage.
Use --sample-ratio 1.0 for a full run (e.g. final submission).

Input:
    data/qa_pairs.json
    results/ablation_results.json

Output:
    results/evaluation_results.json   ← atomic write
"""

import argparse
import json
import logging
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging — stdout + file
# ---------------------------------------------------------------------------
_log_dir = Path("logs")
_log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_dir / "evaluation.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("Evaluation")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
QA_PATH       = Path("data/qa_pairs.json")
ABLATION_PATH = Path("results/ablation_results.json")
OUTPUT_PATH   = Path("results/evaluation_results.json")

DEFAULT_SAMPLE_RATIO = 0.20   # 20 % of each config for RAGAS
DEFAULT_SEED         = 42


# ---------------------------------------------------------------------------
# Atomic JSON write
# ---------------------------------------------------------------------------

RAGAS_JUDGE_MODEL = "llama3"

def _ollama_stop(model_name: str) -> None:
    """Evict a model from Ollama's memory to free VRAM."""
    try:
        import subprocess
        subprocess.run(["ollama", "stop", model_name], capture_output=True, check=False)
        logger.debug("Evicted %s from memory.", model_name)
    except Exception as e:
        logger.debug("Could not stop ollama model: %s", e)

def _wait_for_memory(seconds: int) -> None:
    import time
    time.sleep(seconds)

def _atomic_json_write(path: Path, data: Any) -> None:
    """Write JSON atomically via .tmp → rename (POSIX atomic)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_WS_RE    = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Lexical metrics
# ---------------------------------------------------------------------------

def exact_match(pred: str, gt: str) -> float:
    return float(normalize(pred) == normalize(gt))


def token_f1(pred: str, gt: str) -> float:
    pred_tokens = normalize(pred).split()
    gt_tokens   = normalize(gt).split()

    if not pred_tokens or not gt_tokens:
        return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_success(pred: str) -> float:
    """1.0 if the answer is non-empty and not an [ERROR] sentinel."""
    if not pred:
        return 0.0
    if pred.lower().startswith("[error]"):
        return 0.0
    if pred == "[NO ANSWER]":
        return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# RAGAS semantic metrics
# ---------------------------------------------------------------------------

def try_ragas(
    queries:       list[str],
    answers:       list[str],
    contexts:      list[list[str]],
    ground_truths: list[str],
) -> dict[str, float]:
    """
    Run RAGAS faithfulness / answer_relevancy / context_recall.

    Returns an empty dict on any failure so lexical metrics are never
    blocked by a RAGAS import or runtime error.

    API notes
    ─────────
    ragas.run_config.RunConfig is imported explicitly here — it was
    previously used but never imported, causing a NameError.
    The RunConfig object is passed to evaluate() as `run_config=`
    (keyword argument), NOT as a dict.
    """
    if not queries:
        logger.warning("try_ragas called with empty query list — skipping.")
        return {}

    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_recall,
            faithfulness,
        )
        from ragas.run_config import RunConfig          # ← THE FIX: explicit import
        from langchain_ollama import ChatOllama, OllamaEmbeddings

        dataset = Dataset.from_dict({
            "question":  queries,
            "answer":    answers,
            "contexts":  contexts,
            "reference": ground_truths,
        })

        llm        = ChatOllama(model=RAGAS_JUDGE_MODEL)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # RunConfig object — passed as kwarg, not a raw dict
        run_cfg = RunConfig(
            timeout=180,      # seconds per RAGAS call
            max_workers=1,    # serial: avoids Ollama rate-limit collisions
        )

        result = ragas_evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
            ],
            llm=llm,
            embeddings=embeddings,
            run_config=run_cfg,     # ← correct: RunConfig object, not dict
        )

        scores = result.to_pandas().mean(numeric_only=True).to_dict()
        logger.info("RAGAS scores: %s", scores)

        return {
            "faithfulness":     round(float(scores.get("faithfulness",     0.0)), 4),
            "answer_relevance": round(float(scores.get("answer_relevancy", 0.0)), 4),
            "context_recall":   round(float(scores.get("context_recall",   0.0)), 4),
        }

    except Exception as exc:
        logger.error("RAGAS evaluation failed: %s", exc, exc_info=True)
        return {}

    finally:
        # Evict judge so next config's pipeline can load cleanly
        import subprocess
        subprocess.run(["ollama", "stop", "llama3"], capture_output=True, check=False)
        import time
        time.sleep(4)


# ---------------------------------------------------------------------------
# Stratified sampler
# ---------------------------------------------------------------------------

def stratified_sample(
    items:        list[dict[str, Any]],
    sample_ratio: float,
    seed:         int,
) -> list[dict[str, Any]]:
    """
    Return a random sample of `items` of size ceil(len * sample_ratio).

    Guarantees at least 1 item is returned even for very small configs.
    Seeded for reproducibility across evaluation runs.
    """
    if sample_ratio >= 1.0:
        return items

    rng      = random.Random(seed)
    n_sample = max(1, round(len(items) * sample_ratio))
    sampled  = rng.sample(items, min(n_sample, len(items)))

    logger.info(
        "  Sampled %d / %d items (ratio=%.0f%%).",
        len(sampled), len(items), sample_ratio * 100,
    )
    return sampled


# ---------------------------------------------------------------------------
# Single-config evaluator
# ---------------------------------------------------------------------------

def evaluate_config(
    config_name:  str,
    outputs:      list[dict[str, Any]],
    qa_lookup:    dict[str, list[str]],
    sample_ratio: float,
    seed:         int,
) -> dict[str, Any]:
    """
    Score one ablation config.

    Lexical metrics run on ALL outputs.
    RAGAS runs on a stratified sample of size (sample_ratio * len(outputs)).

    Args:
        config_name:  e.g. "full_adaptive_b1.0"
        outputs:      List of per-query result dicts from ablation_runner.
        qa_lookup:    question → [ground_truth, …] mapping.
        sample_ratio: Fraction of outputs to pass to RAGAS.
        seed:         RNG seed for reproducible sampling.

    Returns:
        Dict of all metric scores for this config.
    """
    em_list, f1_list, success_list = [], [], []

    for item in outputs:
        q    = item.get("query", "").strip()
        pred = item.get("answer", "")

        gt_list = qa_lookup.get(q, [""])
        gt      = gt_list[0]

        em_list.append(exact_match(pred, gt))
        f1_list.append(token_f1(pred, gt))
        success_list.append(answer_success(pred))

    n = len(outputs)
    result: dict[str, Any] = {
        "n_queries":           n,
        "exact_match":         round(sum(em_list)      / n, 4) if n else 0.0,
        "token_f1":            round(sum(f1_list)      / n, 4) if n else 0.0,
        "answer_success_rate": round(sum(success_list) / n, 4) if n else 0.0,
    }

    # ── RAGAS on sampled subset ───────────────────────────────────────
    sampled = stratified_sample(outputs, sample_ratio, seed)

    ragas_queries:  list[str]       = []
    ragas_answers:  list[str]       = []
    ragas_contexts: list[list[str]] = []
    ragas_gts:      list[str]       = []

    for item in sampled:
        q    = item.get("query", "").strip()
        pred = item.get("answer", "")
        gt   = (qa_lookup.get(q, [""]))[0]

        ragas_queries.append(q)
        ragas_answers.append(pred)
        ragas_contexts.append(item.get("retrieved_texts", [pred]) or [pred])
        ragas_gts.append(gt)

    logger.info(
        "  Running RAGAS on %d items for config '%s' …",
        len(ragas_queries), config_name,
    )
    ragas_scores = try_ragas(ragas_queries, ragas_answers, ragas_contexts, ragas_gts)

    result["ragas_sample_size"] = len(sampled)
    result.update(ragas_scores)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(
    qa_path:       Path = QA_PATH,
    ablation_path: Path = ABLATION_PATH,
    output_path:   Path = OUTPUT_PATH,
    sample_ratio:  float = DEFAULT_SAMPLE_RATIO,
    seed:          int   = DEFAULT_SEED,
) -> dict[str, Any]:
    """
    Evaluate all ablation configs and write results atomically.

    Returns:
        Final results dict (also written to output_path).
    """
    print("=" * 60)
    print("  Evaluation — Ablation Study Scoring")
    print("=" * 60)
    print(f"  QA path       : {qa_path}")
    print(f"  Ablation path : {ablation_path}")
    print(f"  Output path   : {output_path}")
    print(f"  RAGAS sample  : {sample_ratio * 100:.0f}%")
    print(f"  Seed          : {seed}")
    print()

    # ── Load inputs ──────────────────────────────────────────────────
    if not qa_path.exists():
        raise FileNotFoundError(
            f"QA dataset not found at '{qa_path}'. Run load_qasper.py first."
        )
    if not ablation_path.exists():
        raise FileNotFoundError(
            f"Ablation results not found at '{ablation_path}'. "
            f"Run ablation_runner.py first."
        )

    with open(qa_path, encoding="utf-8") as f:
        qa_pairs: list[dict[str, Any]] = json.load(f)

    with open(ablation_path, encoding="utf-8") as f:
        ablations: dict[str, list[dict[str, Any]]] = json.load(f)

    # Build lookup: question → [ground_truth, …]  (handles duplicates)
    qa_lookup: dict[str, list[str]] = {}
    for item in qa_pairs:
        q = item["question"].strip()
        a = item.get("ground_truth_answer", "").strip()
        qa_lookup.setdefault(q, []).append(a)

    logger.info("QA lookup built: %d unique questions.", len(qa_lookup))
    logger.info("Ablation configs found: %s", list(ablations.keys()))

    # ── Evaluate each config ─────────────────────────────────────────
    final_results: dict[str, Any] = {}

    for config_name, outputs in ablations.items():
        logger.info("Evaluating config: '%s' (%d queries) …", config_name, len(outputs))

        config_result = evaluate_config(
            config_name  = config_name,
            outputs      = outputs,
            qa_lookup    = qa_lookup,
            sample_ratio = sample_ratio,
            seed         = seed,
        )
        final_results[config_name] = config_result

        logger.info(
            "  EM=%.4f  F1=%.4f  Success=%.4f  "
            "Faith=%.4f  Rel=%.4f  CtxRec=%.4f",
            config_result.get("exact_match",         0.0),
            config_result.get("token_f1",            0.0),
            config_result.get("answer_success_rate", 0.0),
            config_result.get("faithfulness",        0.0),
            config_result.get("answer_relevance",    0.0),
            config_result.get("context_recall",      0.0),
        )

    # ── Atomic save ───────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_ratio": sample_ratio,
        "seed":         seed,
        "configs":      final_results,
    }
    _atomic_json_write(output_path, payload)

    # ── Console summary ───────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  {'Config':<28} {'EM':>6} {'F1':>6} {'Succ':>6} {'Faith':>7} {'Rel':>7} {'CtxR':>7}")
    print("─" * 60)

    for cfg, v in final_results.items():
        print(
            f"  {cfg:<28}"
            f"  {v.get('exact_match',         0.0):>5.3f}"
            f"  {v.get('token_f1',            0.0):>5.3f}"
            f"  {v.get('answer_success_rate', 0.0):>5.3f}"
            f"  {v.get('faithfulness',        0.0):>6.3f}"
            f"  {v.get('answer_relevance',    0.0):>6.3f}"
            f"  {v.get('context_recall',      0.0):>6.3f}"
        )

    print("─" * 60)
    print(f"\n  Saved → {output_path}")

    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score ablation_results.json with lexical + RAGAS metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--qa-path", type=Path, default=QA_PATH,
        help="Path to QA pairs JSON.",
    )
    parser.add_argument(
        "--ablation-path", type=Path, default=ABLATION_PATH,
        help="Path to ablation_results.json.",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help="Path to write evaluation_results.json.",
    )
    parser.add_argument(
        "--sample-ratio", type=float, default=DEFAULT_SAMPLE_RATIO,
        help="Fraction of each config to pass to RAGAS (0.0–1.0). "
             "Use 1.0 for a full run.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="RNG seed for reproducible sampling.",
    )
    args = parser.parse_args()

    evaluate(
        qa_path       = args.qa_path,
        ablation_path = args.ablation_path,
        output_path   = args.output,
        sample_ratio  = args.sample_ratio,
        seed          = args.seed,
    )