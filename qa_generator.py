"""
qa_generator.py — LLM-Assisted QA Pair Generation
Owner: Samved Jain

Generates the 30 manually-authored QA pairs (10 factual, 10 conceptual,
10 complex) required for the evaluation set, using the local Ollama LLM
to produce questions and answers from QASPER chunks.

Output schema (qa_pairs.json entry):
{
    "question":            str,
    "ground_truth_answer": str,
    "query_type":          "factual" | "conceptual" | "complex",
    "source_document":     str,
    "relevant_chunk_ids":  [str]
}
"""

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
LLM_MODEL = "llama3.2:3b"

# Target counts per type for the 30 supplementary pairs
TARGET_COUNTS: dict[str, int] = {
    "factual": 10,
    "conceptual": 10,
    "complex": 10,
}

# How many times to retry a failing LLM call before skipping a chunk
MAX_RETRIES = 3
RETRY_DELAY_SEC = 2


# ---------------------------------------------------------------------------
# Prompt templates (from spec, slightly expanded for reliability)
# ---------------------------------------------------------------------------

def _build_prompt(chunk_text: str, query_type: str) -> str:
    """
    Build the generation prompt for a given query type.

    The prompt is deliberately strict about JSON-only output to make
    parsing reliable.  It also reminds the LLM of what each type means
    so classification stays consistent.

    Args:
        chunk_text:  The source chunk text.
        query_type:  "factual" | "conceptual" | "complex"

    Returns:
        Formatted prompt string.
    """
    type_guidance = {
        "factual": (
            "A FACTUAL question has a short, specific answer: a name, number, "
            "date, or single entity directly stated in the text.  "
            "Examples: 'What dataset was used?', 'How many layers does the model have?'"
        ),
        "conceptual": (
            "A CONCEPTUAL question requires explanation or reasoning about ideas "
            "in the text.  The answer should be 2–4 sentences.  "
            "Examples: 'Why does the model use attention?', 'How does the training procedure work?'"
        ),
        "complex": (
            "A COMPLEX question requires connecting or comparing multiple pieces "
            "of information — either within this chunk or implying cross-document "
            "reasoning.  The answer should be 3–6 sentences.  "
            "Examples: 'What are the trade-offs between X and Y?', "
            "'Analyze the impact of Z on downstream performance.'"
        ),
    }

    return f"""You are generating evaluation data for a RAG system.
Given the document chunk below, generate exactly one {query_type.upper()} question and its answer.

Query type definition:
{type_guidance[query_type]}

IMPORTANT RULES:
- The question must be answerable SOLELY from the provided chunk.
- The answer must be grounded in the chunk text — do not hallucinate.
- Respond in VALID JSON only. No preamble, no markdown, no explanation.
- Use this exact structure:
{{
    "question": "...",
    "answer": "...",
    "query_type": "{query_type}"
}}

Document chunk:
\"\"\"
{chunk_text}
\"\"\"

JSON response:"""


# ---------------------------------------------------------------------------
# QAGenerator
# ---------------------------------------------------------------------------

class QAGenerator:
    """
    Generates QA pairs from document chunks using a local Ollama LLM.

    Workflow:
        1. Accept a list of chunks (from ingestion.py).
        2. For each target query type, randomly sample chunks.
        3. Call the LLM with the typed prompt.
        4. Parse, validate, and store the output.
        5. Save all pairs to data/qa_pairs.json.
    """

    def __init__(self, llm_model: str = LLM_MODEL):
        """
        Args:
            llm_model: Ollama model identifier (must be pulled already).
        """
        self.llm_model = llm_model
        self.generated_pairs: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        target_counts: dict[str, int] = TARGET_COUNTS,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """
        Generate QA pairs by sampling chunks and prompting the LLM.

        Args:
            chunks:        Full chunk list from ingestion.py.
            target_counts: How many pairs to generate per query type.
            seed:          Random seed for reproducible chunk sampling.

        Returns:
            List of QA pair dicts conforming to the agreed schema.
        """
        random.seed(seed)
        self.generated_pairs = []

        # Shuffle once so different query types sample different chunks
        shuffled = chunks.copy()
        random.shuffle(shuffled)

        # Partition the shuffled list into thirds (one per type)
        # so the same chunk doesn't appear in multiple qa pairs
        n = len(shuffled)
        third = n // 3
        type_chunk_pools: dict[str, List[Dict[str, Any]]] = {
            "factual":    shuffled[:third],
            "conceptual": shuffled[third:2*third],
            "complex":    shuffled[2*third:],
        }

        for query_type, count in target_counts.items():
            pool = type_chunk_pools[query_type]
            if len(pool) < count:
                print(f"[QAGenerator] Warning: not enough chunks for {query_type}. "
                      f"Need {count}, have {len(pool)}.")
                count = len(pool)

            selected_chunks = random.sample(pool, count)
            print(f"\n[QAGenerator] Generating {count} {query_type} QA pairs …")

            for chunk in tqdm(selected_chunks, desc=query_type):
                pair = self._generate_single(chunk, query_type)
                if pair is not None:
                    self.generated_pairs.append(pair)

        print(f"\n[QAGenerator] Generated {len(self.generated_pairs)} pairs total.")
        return self.generated_pairs

    def _generate_single(
        self,
        chunk: Dict[str, Any],
        query_type: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Call the LLM to generate one QA pair from a single chunk.

        Retries up to MAX_RETRIES times on parse failure before skipping.

        Args:
            chunk:      Chunk dict (must have chunk_id, text, source).
            query_type: "factual" | "conceptual" | "complex"

        Returns:
            QA pair dict conforming to the spec schema, or None on failure.
        """
        prompt = _build_prompt(chunk["text"], query_type)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = ollama.chat(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.3},   # low temp for factual accuracy
                )
                raw_text = response["message"]["content"].strip()
                parsed = self._parse_llm_response(raw_text, query_type)

                if parsed is not None:
                    # Attach provenance metadata (spec schema)
                    parsed["source_document"] = chunk["source"]
                    parsed["relevant_chunk_ids"] = [chunk["chunk_id"]]
                    # Rename "answer" → "ground_truth_answer" (spec key)
                    parsed["ground_truth_answer"] = parsed.pop("answer")
                    return parsed

            except Exception as e:
                print(f"[QAGenerator] Attempt {attempt}/{MAX_RETRIES} failed for "
                      f"chunk {chunk['chunk_id']}: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SEC)

        print(f"[QAGenerator] Skipping chunk {chunk['chunk_id']} after {MAX_RETRIES} failures.")
        return None

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_llm_response(
        raw_text: str, expected_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parse and validate the LLM's JSON response.

        The LLM sometimes wraps JSON in markdown fences or adds a preamble.
        This method strips those and validates the required keys.

        Args:
            raw_text:      Raw string from the LLM.
            expected_type: The query_type we asked the LLM to generate.

        Returns:
            Parsed dict with keys: question, answer, query_type
            Returns None if parsing or validation fails.
        """
        # Strip markdown code fences if present (```json ... ```)
        clean = re.sub(r"```(?:json)?|```", "", raw_text).strip()

        # Find the first { ... } block (handles extra text before/after JSON)
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            print(f"[QAGenerator] No JSON object found in response: {raw_text[:100]}")
            return None

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as e:
            print(f"[QAGenerator] JSON parse error: {e}")
            return None

        # Validate required keys
        required = {"question", "answer", "query_type"}
        if not required.issubset(data.keys()):
            print(f"[QAGenerator] Missing keys. Got: {list(data.keys())}")
            return None

        # Validate query_type matches what we asked for (LLM sometimes ignores this)
        if data["query_type"] not in ("factual", "conceptual", "complex"):
            data["query_type"] = expected_type   # override if invalid

        # Sanity check — non-empty strings
        if not data["question"].strip() or not data["answer"].strip():
            print("[QAGenerator] Empty question or answer.")
            return None

        return data

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path = QA_OUTPUT_PATH) -> None:
        """
        Save all generated QA pairs to JSON.

        If the file already exists (e.g., QASPER pairs were loaded first),
        this *appends* the new pairs to the existing list.

        Args:
            path: Output file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        existing: List[Dict[str, Any]] = []
        if path.exists():
            with open(path, "r") as f:
                existing = json.load(f)
            print(f"[QAGenerator] Found {len(existing)} existing pairs. Appending.")

        combined = existing + self.generated_pairs
        with open(path, "w") as f:
            json.dump(combined, f, indent=2)

        print(f"[QAGenerator] Saved {len(combined)} total QA pairs → {path}")

    def load_existing(self, path: Path = QA_OUTPUT_PATH) -> List[Dict[str, Any]]:
        """
        Load already-generated QA pairs from disk.

        Args:
            path: Path to qa_pairs.json.

        Returns:
            List of QA pair dicts.
        """
        if not path.exists():
            raise FileNotFoundError(f"QA pairs file not found: {path}")

        with open(path, "r") as f:
            pairs = json.load(f)

        print(f"[QAGenerator] Loaded {len(pairs)} QA pairs from {path}")
        return pairs

    # ------------------------------------------------------------------
    # Statistics helper
    # ------------------------------------------------------------------

    def print_stats(self) -> None:
        """Print a summary of generated pairs by query type."""
        from collections import Counter
        counts = Counter(p["query_type"] for p in self.generated_pairs)
        print("\n--- QA Pair Generation Summary ---")
        for qtype in ("factual", "conceptual", "complex"):
            print(f"  {qtype:<12}: {counts.get(qtype, 0)}")
        print(f"  {'TOTAL':<12}: {len(self.generated_pairs)}")


# ---------------------------------------------------------------------------
# Smoke test (dry run with a mock LLM response)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import unittest.mock as mock

    # Simulate what the LLM would return
    mock_response = {
        "message": {
            "content": json.dumps({
                "question": "What dataset was used to evaluate the model?",
                "answer": "The model was evaluated on the SQuAD v1.1 dataset.",
                "query_type": "factual",
            })
        }
    }

    dummy_chunks = [
        {
            "chunk_id": f"doc1_chunk_{i:03d}",
            "text": f"The model was trained on SQuAD. It achieved 91.2 F1. Chunk {i}.",
            "source": "paper1.pdf",
            "position": i,
            "score": 0.0,
        }
        for i in range(35)   # enough for 30 pairs with some headroom
    ]

    generator = QAGenerator()

    with mock.patch("ollama.chat", return_value=mock_response):
        pairs = generator.generate_from_chunks(
            dummy_chunks,
            target_counts={"factual": 2, "conceptual": 2, "complex": 2},
        )

    generator.print_stats()
    print("\n--- First generated pair ---")
    print(json.dumps(pairs[0], indent=2))