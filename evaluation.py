import json
import logging
import re
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Evaluation")

QA_PATH       = Path("data/qa_pairs.json")
ABLATION_PATH = Path("results/ablation_results.json")
OUTPUT_PATH   = Path("results/evaluation_results.json")

# Single model for both generation and RAGAS judge — evicted between uses
PIPELINE_MODEL    = "phi3:mini"
RAGAS_JUDGE_MODEL = "phi3:mini"
RAGAS_EMBED_MODEL = "nomic-embed-text"


# ---------------------------------------------------------
# OLLAMA MEMORY MANAGEMENT
# ---------------------------------------------------------

def _ollama_stop(model: str) -> None:
    """Unload a model from Ollama RAM. Safe to call if not loaded."""
    try:
        subprocess.run(
            ["ollama", "stop", model],
            capture_output=True, encoding="utf-8",
            errors="replace", timeout=30,
        )
        logger.info("Evicted from Ollama: %s", model)
    except Exception as e:
        logger.warning("ollama stop %s failed: %s", model, e)


def _wait_for_memory(seconds: int = 8) -> None:
    """Give Ollama time to fully release memory after a stop call."""
    logger.info("Waiting %ds for Ollama to release memory…", seconds)
    time.sleep(seconds)


def _ollama_pull_if_missing(model: str) -> bool:
    """Pull model if not present locally. Returns True if available."""
    try:
        check = subprocess.run(
            ["ollama", "show", model],
            capture_output=True, encoding="utf-8",
            errors="replace", timeout=15,
        )
        if check.returncode == 0:
            return True
        logger.info("Pulling '%s'…", model)
        pull = subprocess.run(
            ["ollama", "pull", model],
            encoding="utf-8", errors="replace", timeout=600,
        )
        return pull.returncode == 0
    except Exception as e:
        logger.error("Pull failed for %s: %s", model, e)
        return False


# ---------------------------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------------------------

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def exact_match(pred, gt):
    return float(normalize(pred) == normalize(gt))


def token_f1(pred, gt):
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


def answer_success(pred):
    if not pred:
        return 0.0
    if pred.lower().startswith("[error]"):
        return 0.0
    if pred == "[NO ANSWER]":
        return 0.0
    return 1.0


# ---------------------------------------------------------
# SAFE RAGAS
# ---------------------------------------------------------

def try_ragas(queries, answers, contexts, ground_truths):
    """
    Run RAGAS semantic metrics with Ollama memory safety.

    Memory strategy:
      - Evict phi3:mini (pipeline model) before RAGAS loads it as judge.
        Both use the same model tag but Ollama treats each ChatOllama()
        instance as a separate load — without eviction, two copies sit in
        RAM simultaneously and cause HTTP 500 OOM errors.
      - keep_alive="0m" on ChatOllama: model unloads immediately after
        each call instead of staying hot for 5 minutes. This prevents
        memory from accumulating across the 90 sequential RAGAS calls.
      - max_workers=1: strictly sequential — no parallel LLM calls that
        would queue up and time out waiting for Ollama on CPU.
      - timeout=300: 5 min per call; phi3:mini on CPU needs ~60-90s.
      - tinyllama was tried but fails RagasOutputParserException because
        it can't reliably emit the structured JSON RAGAS requires.
        phi3:mini is the minimum capable model for RAGAS judging locally.
    """
    try:
        from ragas import evaluate, EvaluationDataset, RunConfig
        from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextRecall
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_ollama import ChatOllama, OllamaEmbeddings

        # Evict the pipeline model so RAGAS can reload it cleanly
        _ollama_stop(PIPELINE_MODEL)
        _wait_for_memory(seconds=8)

        if not _ollama_pull_if_missing(RAGAS_JUDGE_MODEL):
            raise RuntimeError(f"Judge model '{RAGAS_JUDGE_MODEL}' unavailable.")
        if not _ollama_pull_if_missing(RAGAS_EMBED_MODEL):
            raise RuntimeError(f"Embedding model '{RAGAS_EMBED_MODEL}' unavailable.")

        # keep_alive="0m" → model unloads after each call, preventing
        # two concurrent copies in RAM between sequential RAGAS jobs
        llm = LangchainLLMWrapper(
            ChatOllama(model=RAGAS_JUDGE_MODEL, temperature=0, keep_alive="0m")
        )
        embeddings = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(model=RAGAS_EMBED_MODEL, keep_alive="0m")
        )

        samples = [
            {
                "user_input":         q,
                "response":           a,
                "retrieved_contexts": c if isinstance(c, list) else [c],
                "reference":          g,
            }
            for q, a, c, g in zip(queries, answers, contexts, ground_truths)
        ]
        dataset = EvaluationDataset.from_list(samples)

        run_config = RunConfig(
            timeout=300,      # 5 min per LLM call
            max_retries=3,    # phi3:mini occasionally mis-formats JSON; retry
            max_wait=60,
            max_workers=1,    # sequential — never overwhelm Ollama on CPU
        )

        logger.info(
            "Running RAGAS | judge=%s embed=%s n=%d",
            RAGAS_JUDGE_MODEL, RAGAS_EMBED_MODEL, len(queries),
        )
        result = evaluate(
            dataset,
            metrics=[
                Faithfulness(llm=llm),
                ResponseRelevancy(llm=llm, embeddings=embeddings),
                LLMContextRecall(llm=llm),
            ],
            run_config=run_config,
        )

        scores = result.to_pandas().mean(numeric_only=True).to_dict()
        logger.info("RAGAS scores: %s", scores)

        return {
            "faithfulness":     round(float(scores.get("faithfulness",       0)), 4),
            "answer_relevance": round(float(scores.get("response_relevancy", 0)), 4),
            "context_recall":   round(float(scores.get("context_recall",     0)), 4),
        }

    except Exception as e:
        logger.error("RAGAS failed: %s", e)
        return {}

    finally:
        # Evict judge so next config's pipeline can load cleanly
        _ollama_stop(RAGAS_JUDGE_MODEL)
        _wait_for_memory(seconds=4)


# ---------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------

def evaluate():
    print("=" * 50)
    print("EVALUATION")
    print("=" * 50)

    with open(QA_PATH, encoding="utf-8") as f:
        qa_pairs = json.load(f)

    with open(ABLATION_PATH, encoding="utf-8") as f:
        ablations = json.load(f)

    qa_lookup: dict = {}
    for item in qa_pairs:
        q = item["question"].strip()
        a = item["ground_truth_answer"].strip()
        if q not in qa_lookup:
            qa_lookup[q] = []
        qa_lookup[q].append(a)

    logger.info("QA lookup size: %d", len(qa_lookup))

    final_results: dict = {}

    for config, outputs in ablations.items():
        logger.info("Evaluating config: %s  (%d queries)", config, len(outputs))

        em_list, f1_list, success_list = [], [], []
        queries, answers, contexts, gts = [], [], [], []

        for item in outputs:
            q    = item["query"]
            pred = item["answer"]

            gt_list = qa_lookup.get(q.strip(), [""])
            gt      = gt_list[0]

            em_list.append(exact_match(pred, gt))
            f1_list.append(token_f1(pred, gt))
            success_list.append(answer_success(pred))

            queries.append(q)
            answers.append(pred)
            contexts.append(item.get("retrieved_texts", [pred]))
            gts.append(gt)

        n = len(outputs)

        result = {
            "exact_match":         sum(em_list) / n,
            "token_f1":            sum(f1_list) / n,
            "answer_success_rate": sum(success_list) / n,
        }

        ragas_scores = try_ragas(queries, answers, contexts, gts)
        result.update(ragas_scores)

        final_results[config] = result
        logger.info("Config %s done: %s", config, result)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print("\nRESULTS:")
    for k, v in final_results.items():
        print(f"  {k}: {v}")
    print(f"\nSaved → {OUTPUT_PATH}")


# ---------------------------------------------------------

if __name__ == "__main__":
    evaluate()