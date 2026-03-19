import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Evaluation")

QA_PATH = Path("data/qa_pairs.json")
ABLATION_PATH = Path("results/ablation_results.json")
OUTPUT_PATH = Path("results/evaluation_results.json")


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
    gt_tokens = normalize(gt).split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)


def answer_success(pred):
    if not pred:
        return 0.0
    if pred.lower().startswith("[error]"):
        return 0.0
    return 1.0


# ---------------------------------------------------------
# SAFE RAGAS (NO CRASH VERSION)
# ---------------------------------------------------------

def try_ragas(queries, answers, contexts, ground_truths):
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics.collections import (
            faithfulness,
            answer_relevancy,
            context_recall,
        )
        from langchain_ollama import ChatOllama, OllamaEmbeddings

        dataset = Dataset.from_dict({
            "question": queries,
            "answer": answers,
            "contexts": contexts,
            "reference": ground_truths,
        })

        llm = ChatOllama(model="phi3:mini")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        result = evaluate(
            dataset,
            metrics=[
                faithfulness(),
                answer_relevancy(),
                context_recall(),
            ],
            llm=llm,
            embeddings=embeddings,
        )

        scores = result.to_pandas().mean().to_dict()

        return {
            "faithfulness": round(scores.get("faithfulness", 0), 4),
            "answer_relevance": round(scores.get("answer_relevancy", 0), 4),
            "context_recall": round(scores.get("context_recall", 0), 4),
        }

    except Exception as e:
        logger.error(f"RAGAS failed: {e}")
        return {}


# ---------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------

def evaluate():

    print("=" * 50)
    print("EVALUATION")
    print("=" * 50)

    with open(QA_PATH) as f:
        qa_pairs = json.load(f)

    with open(ABLATION_PATH) as f:
        ablations = json.load(f)

    # FIX: allow duplicate questions → use list mapping
    qa_lookup = {}
    for item in qa_pairs:
        q = item["question"].strip()
        a = item["ground_truth_answer"].strip()

        if q not in qa_lookup:
            qa_lookup[q] = []
        qa_lookup[q].append(a)

    logger.info(f"QA lookup size: {len(qa_lookup)}")

    final_results = {}

    for config, outputs in ablations.items():
        logger.info(f"Evaluating: {config}")

        em_list, f1_list, success_list = [], [], []
        queries, answers, contexts, gts = [], [], [], []

        for item in outputs:
            q = item["query"]
            pred = item["answer"]

            # handle duplicate GTs → take first
            gt_list = qa_lookup.get(q, [""])
            gt = gt_list[0]

            em = exact_match(pred, gt)
            f1 = token_f1(pred, gt)
            success = answer_success(pred)

            em_list.append(em)
            f1_list.append(f1)
            success_list.append(success)

            queries.append(q)
            answers.append(pred)
            contexts.append(item.get("retrieved_texts", [pred]))
            gts.append(gt)

        n = len(outputs)

        result = {
            "exact_match": sum(em_list) / n,
            "token_f1": sum(f1_list) / n,
            "answer_success_rate": sum(success_list) / n,
        }

        # try semantic metrics
        ragas_scores = try_ragas(queries, answers, contexts, gts)
        result.update(ragas_scores)

        final_results[config] = result

    # save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(final_results, f, indent=2)

    print("\nRESULTS:")
    for k, v in final_results.items():
        print(k, v)


# ---------------------------------------------------------

if __name__ == "__main__":
    evaluate()