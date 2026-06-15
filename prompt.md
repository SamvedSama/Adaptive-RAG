# Eco-RAG (Resource-Aware Adaptive RAG) - Project Context

## 1. Project Goal & Philosophy
The objective of this project is to build **Eco-RAG**, a resource-aware Adaptive Retrieval-Augmented Generation system. 

Traditional Adaptive RAG systems are "environment-blind"—they use massive, slow LLM routers to statically evaluate the complexity of a user's question and default to the most rigorous retrieval method necessary to answer it. Under heavy API payloads or resource exhaustion, this causes runaway costs and system timeouts.

**Eco-RAG** solves this through "Graceful Degradation". We replace the slow GenAI router with a blazingly fast, lightweight Machine Learning classifier. This Micro-Router makes decisions based on TWO inputs:
1. **The Query** (Text semantic embeddings)
2. **The Budget** (A numerical metric from 0.0 to 1.0 representing system health, compute availability, or API credit limits)

Based on the Budget constraint, the router dynamically shifts its decision boundaries. It compromises on retrieval depth to guarantee uptime—defaulting to "Direct LLM" or fast "BM25" when stressed, and opening up to full "Multi-Hop FAISS & Reranking" when resources are optimal. The entire system is built to run totally locally within an 8GB VRAM constraint constraints.

---

## 2. Dynamic Routing Logic (The Core)
The router outputs one of three paths:
* **`Multi_Hop_FAISS` (High Budget: 0.8 - 1.0):** Optimal path. Uses Sentence-Transformers + FAISS dense retrieval + Cross-Encoder Reranking for complex questions.
* **`Single_Hop_BM25` (Medium Budget: 0.4 - 0.7):** Stressed path. Penalizes expensive multi-hop logic. Drops to fast, sparse keyword-based Rank-BM25 retrieval to save compute.
* **`Direct_LLM` (Low Budget: 0.0 - 0.3):** Failsafe path. System is overloaded. Bypasses retrieval entirely and directly queries the LLM to guarantee an immediate, low-fidelity response over an outright crash.

---

## 3. File-by-File Architecture Breakdown

### 🧠 The Brain
* **`router.py`**
  * **Role:** The budget-aware Machine Learning Micro-Router.
  * **Mechanism:** Converts the query into dense vector features using `sentence-transformers/all-MiniLM-L6-v2`. It takes this embedding, horizontally appends the numerical `[budget]` metric onto the vector array, and feeds it into a trained `RandomForestClassifier` (sklearn).
  * **Why it matters:** Millisecond-level routing latency. Shifts architectural paths organically based on system stress rather than just query complexity.

* **`train_router.py`**
  * **Role:** Generates the synthetic dataset and trains the RandomForest model via scikit-learn. Assigns proper route labels based on simulated query complexity vs. injected budget combinations.

### ⚙️ The Orchestration Engine
* **`adaptive_pipeline.py`**
  * **Role:** The core traffic conductor.
  * **Mechanism:** Receives the user `query` and system `budget`. Calls `router.py` to get the routing decision, then executes the corresponding retrieval paths. Finally, passes the context and query to the local LLM via a subprocess call to Ollama.
  * **Configurations:** Supports dynamic switching between ablation configs (Full Adaptive, Router Only, Reranker Only, Naive RAG).

### 🔍 The Retrieval Layer
* **`ingestion.py`** & **`load_qasper.py`**
  * **Role:** Downloads the QASPER dataset (AI/NLP papers), parses PDFs via PyMuPDF, cleans text (handling tricky Unicode), and chunks into overlapped segments.
* **`bm25_retriever.py`**
  * **Role:** Executes the lightweight, sparse keyword search (`Single_Hop_BM25`).
* **`faiss_retriever.py`**
  * **Role:** Executes the dense semantic vector search running in memory. 
* **`hybrid_retriever.py`**
  * **Role:** Fuses BM25 and FAISS results using Reciprocal Rank Fusion (RRF) for the highest fidelity `Multi_Hop` retrieval path.
* **`reranker.py`**
  * **Role:** Utilizes a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to re-score and prune candidate chunks for maximum precision.

### 📊 Frontend & Evaluation
* **`app.py`**
  * **Role:** A Streamlit interactive dashboard. Features a slider manually controlling the "System Budget" to visually demonstrate how routing dynamically shifts away from heavy FAISS logic to cheaper fallbacks as you slide the budget toward 0.0.
* **`evaluation.py`, `ablation_runner.py`, `pareto_curve.py`, `latency_tracker.py`**
  * **Role:** Proves the Eco-RAG concept. Measures per-stage latency differences, computes RAGAS scores (Faithfulness/Answer Relevance), and calculates the Pareto curve outlining the tradeoff between accuracy loss and latency/compute gains when standardizing budget constraints.

---

## 4. Instructions for the LLM
When assisting with this project, ensure you adhere to the following constraints:
1. **Never break the architecture:** Maintain the strict decoupling of the ML Router (`router.py`) from the pipeline orchestrator (`adaptive_pipeline.py`).
2. **Prioritize Performance:** Everything runs locally. Assume max 8GB VRAM limitations. Avoid large LLMs for backend classification.
3. **Budget Awareness:** All modifications to routing or pipeline logic must respect and interact with the `0.0 to 1.0` Budget float variable. Ensure fallback mechanisms strictly adhere to the "Graceful Degradation" philosophy.
4. **Tooling limits:** Use python, Ollama (phi3:mini / Llama3.2 3B), sentence-transformers, FAISS, scikit-learn, and Streamlit. No paid APIs.
