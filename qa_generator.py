import json
import re
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import ollama
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QA_OUTPUT_PATH = Path("data/qa_pairs.json")
LLM_MODEL = "phi3:mini"

TARGET_COUNTS = {
    "factual": 10,
    "conceptual": 10,
    "complex": 10,
}

MAX_RETRIES = 3
RETRY_DELAY_SEC = 2


# ---------------------------------------------------------------------------
# Improved Prompt (🔥 diversity fix)
# ---------------------------------------------------------------------------

def _build_prompt(chunk_text: str, query_type: str) -> str:
    chunk_text = chunk_text[:800]

    return f"""
You are generating HIGH-QUALITY and DIVERSE evaluation questions for a RAG system.

Generate ONE UNIQUE {query_type.upper()} question and answer.

STRICT RULES:
- DO NOT repeat generic questions like "What dataset was used?"
- Make the question SPECIFIC to this chunk
- Each question must be DIFFERENT from typical patterns
- Answer MUST be from the text only
- Output ONLY valid JSON

Format:
{{
  "question": "...",
  "answer": "...",
  "query_type": "{query_type}"
}}

TEXT:
\"\"\"
{chunk_text}
\"\"\"
"""


# ---------------------------------------------------------------------------
# QAGenerator
# ---------------------------------------------------------------------------

class QAGenerator:

    def __init__(self, llm_model: str = LLM_MODEL):
        self.llm_model = llm_model
        self.generated_pairs: List[Dict[str, Any]] = []
        self.seen_questions = set()   # ✅ duplicate tracking

    # ------------------------------------------------------------------

    def generate_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        target_counts: dict = TARGET_COUNTS,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:

        random.seed(seed)
        self.generated_pairs = []
        self.seen_questions = set()

        shuffled = chunks.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        third = n // 3

        type_chunk_pools = {
            "factual": shuffled[:third],
            "conceptual": shuffled[third:2*third],
            "complex": shuffled[2*third:],
        }

        for query_type, count in target_counts.items():
            pool = type_chunk_pools[query_type]
            selected_chunks = random.sample(pool, min(count, len(pool)))

            print(f"\n[QAGenerator] Generating {query_type}...")

            for chunk in tqdm(selected_chunks):

                # 🔁 retry until unique
                pair = None
                for _ in range(5):
                    pair = self._generate_single(chunk, query_type)
                    if pair:
                        break

                if pair:
                    self.generated_pairs.append(pair)

        print(f"\nGenerated {len(self.generated_pairs)} QA pairs")
        return self.generated_pairs

    # ------------------------------------------------------------------

    def _generate_single(self, chunk, query_type):

        prompt = _build_prompt(chunk["text"], query_type)

        for attempt in range(MAX_RETRIES):
            try:
                response = ollama.chat(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "temperature": 0.7,  # 🔥 higher diversity
                        "num_predict": 300,
                    },
                )

                raw = response["message"]["content"].strip()
                parsed = self._parse_llm_response(raw, query_type)

                if parsed:
                    q_norm = parsed["question"].strip().lower()

                    # ❌ skip duplicates
                    if q_norm in self.seen_questions:
                        return None

                    self.seen_questions.add(q_norm)

                    parsed["source_document"] = chunk["source"]
                    parsed["relevant_chunk_ids"] = [chunk["chunk_id"]]
                    parsed["ground_truth_answer"] = parsed.pop("answer")

                    return parsed

            except Exception as e:
                print("Error:", e)
                time.sleep(RETRY_DELAY_SEC)

        return None

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_llm_response(raw_text, expected_type):

        clean = re.sub(r"```.*?```", "", raw_text, flags=re.DOTALL).strip()

        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            return None

        try:
            data = json.loads(match.group())
        except:
            return None

        if not all(k in data for k in ["question", "answer", "query_type"]):
            return None

        if not data["question"].strip() or not data["answer"].strip():
            return None

        return data

    # ------------------------------------------------------------------

    def save(self):
        QA_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(QA_OUTPUT_PATH, "w") as f:
            json.dump(self.generated_pairs, f, indent=2)

        print(f"Saved → {QA_OUTPUT_PATH}")

    # ------------------------------------------------------------------

    def print_stats(self):
        from collections import Counter
        c = Counter(p["query_type"] for p in self.generated_pairs)
        print("\nStats:", dict(c))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    CHUNK_PATH = "data/chunks/chunks.json"

    with open(CHUNK_PATH) as f:
        chunks = json.load(f)

    generator = QAGenerator()
    generator.generate_from_chunks(chunks)
    generator.print_stats()
    generator.save()