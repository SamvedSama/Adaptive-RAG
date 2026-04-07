"""
train_router.py — ML Micro-Router Training Pipeline
Owner: Samved Jain

Reads router_training_data.csv, encodes queries into semantic embeddings,
appends the budget metric, trains a RandomForestClassifier, evaluates it,
and exports the model artifact to data/router_model.pkl.

Usage:
    python train_router.py [--csv data/router_training_data.csv]
                           [--output data/router_model.pkl]
                           [--encoder sentence-transformers/all-MiniLM-L6-v2]
                           [--n-estimators 100] [--test-size 0.2] [--seed 42]
                           [--force]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator
from router import StringLabelXGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_router.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

EXPECTED_LABELS = frozenset({"Multi_Hop_FAISS", "Single_Hop_BM25", "Direct_LLM"})
EXPECTED_BUDGETS = frozenset({0.9, 0.5, 0.1})
REQUIRED_COLUMNS = frozenset({"Query_Text", "Budget_Value", "Target_Label"})


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrainConfig:
    csv_path: Path = Path("data/router_training_data.csv")
    model_output_path: Path = Path("data/router_model.pkl")
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    n_estimators: int = 100
    test_size: float = 0.2
    seed: int = 42
    cv_folds: int = 5
    force: bool = False

    def __post_init__(self) -> None:
        if not (0.05 <= self.test_size <= 0.5):
            raise ValueError("test_size must be between 0.05 and 0.5.")
        if self.n_estimators < 10:
            raise ValueError("n_estimators must be ≥ 10.")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be ≥ 2.")


# ── Data Loading & Validation ──────────────────────────────────────────────────

def load_and_validate(cfg: TrainConfig) -> pd.DataFrame:
    """
    Load the training CSV and enforce schema, label, and budget constraints.
    Raises ValueError with a clear message on any violation.
    """
    if not cfg.csv_path.exists():
        raise FileNotFoundError(
            f"Training CSV not found at '{cfg.csv_path}'. Run qa_generator.py first."
        )

    df = pd.read_csv(cfg.csv_path)
    log.info("Loaded %d rows from %s.", len(df), cfg.csv_path)

    # Column check
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    # Null check
    nulls = df[list(REQUIRED_COLUMNS)].isnull().sum()
    if nulls.any():
        raise ValueError(f"CSV contains null values:\n{nulls[nulls > 0]}")

    # Label check
    unknown_labels = set(df["Target_Label"].unique()) - EXPECTED_LABELS
    if unknown_labels:
        raise ValueError(
            f"Unexpected Target_Label values: {unknown_labels}. "
            f"Expected: {EXPECTED_LABELS}"
        )

    # Budget check
    unknown_budgets = set(df["Budget_Value"].round(2).unique()) - EXPECTED_BUDGETS
    if unknown_budgets:
        log.warning("Unexpected Budget_Value entries: %s", unknown_budgets)

    # Empty text check
    empty_queries = (df["Query_Text"].str.strip() == "").sum()
    if empty_queries > 0:
        raise ValueError(f"{empty_queries} rows have empty Query_Text.")

    log.info(
        "Validation passed. Label distribution:\n%s",
        df["Target_Label"].value_counts().to_string(),
    )
    return df


# ── Feature Engineering ────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    encoder: SentenceTransformer,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode query text into L2-normalised sentence embeddings,
    then horizontally concatenate the scalar budget value.

    Returns:
        X: float32 array of shape (N, embedding_dim + 1)
        y: string label array of shape (N,)
    """
    log.info("Encoding %d queries with '%s' ...", len(df), encoder._model_card_data.model_id if hasattr(encoder, '_model_card_data') else "encoder")
    t0 = time.perf_counter()

    embeddings: np.ndarray = encoder.encode(
        df["Query_Text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
    )

    log.info("Encoding complete in %.1fs.", time.perf_counter() - t0)

    budgets = df["Budget_Value"].values.reshape(-1, 1).astype(np.float32)
    X = np.hstack((embeddings.astype(np.float32), budgets))
    y = df["Target_Label"].values

    log.info("Feature matrix shape: %s", X.shape)
    return X, y


# ── Training & Evaluation ──────────────────────────────────────────────────────

def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    cfg: TrainConfig,
    df: "pd.DataFrame | None" = None,
) -> BaseEstimator:
    """
    Split data, train an XGBClassifier, run cross-validation,
    and print a full held-out evaluation report.
    """
    import pandas as pd
    # To prevent data leakage since each query appears multiple times (for different budgets),
    # we MUST split by query group instead of random rows.
    from sklearn.model_selection import GroupShuffleSplit
    if df is not None:
        groups = df["Query_Text"].values
    else:
        # Fallback: treat every row as its own group (no leakage prevention)
        import numpy as np
        groups = np.arange(len(X))
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.seed)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    
    log.info(
        "Split: %d train / %d test (grouped by query text).",
        len(X_train), len(X_test),
    )

    # Compute balanced class weights to fix the Single_Hop_BM25 underrepresentation
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights_train = compute_sample_weight(class_weight="balanced", y=y_train)

    clf = StringLabelXGBClassifier(
        n_estimators=cfg.n_estimators,
        random_state=cfg.seed,
        n_jobs=-1,          # use all available cores
        eval_metric="mlogloss",
        use_label_encoder=False,
    )

    # Cross-validation on training fold
    log.info("Running %d-fold cross-validation on training data ...", cfg.cv_folds)
    from sklearn.model_selection import GroupKFold
    cv = GroupKFold(n_splits=cfg.cv_folds)
    # Note: evaluate on X_train/y_train with groups_train to avoid data leakage
    cv_scores = cross_val_score(clf, X_train, y_train, groups=groups_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    log.info(
        "CV F1-macro: %.4f ± %.4f  (folds: %s)",
        cv_scores.mean(),
        cv_scores.std(),
        [f"{s:.3f}" for s in cv_scores],
    )

    # Final fit on full training fold — with balanced weights
    log.info("Fitting final model on full training fold ...")
    t0 = time.perf_counter()
    clf.fit(X_train, y_train, sample_weight=sample_weights_train)
    log.info("Training complete in %.2fs.", time.perf_counter() - t0)

    # Held-out evaluation
    y_pred = clf.predict(X_test)
    log.info("\nHeld-out Classification Report:\n%s", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=sorted(EXPECTED_LABELS))
    log.info(
        "Confusion Matrix (rows=true, cols=pred, labels=%s):\n%s",
        sorted(EXPECTED_LABELS),
        cm,
    )

    held_out_acc = (y_pred == y_test).mean()
    log.info("Held-out accuracy: %.4f", held_out_acc)

    # Fit on ALL data before saving — with balanced weights for max generalisation
    log.info("Re-fitting on full dataset for final artifact ...")
    sample_weights_full = compute_sample_weight(class_weight="balanced", y=y)
    clf.fit(X, y, sample_weight=sample_weights_full)

    return clf


# ── Model Persistence ──────────────────────────────────────────────────────────

def save_artifact(
    clf: BaseEstimator,
    encoder_name: str,
    feature_dim: int,
    output_path: Path,
) -> None:
    """
    Atomically save the model + provenance metadata as a single joblib artifact.
    Provenance lets router.py verify compatibility at load time.
    """
    artifact = {
        "model": clf,
        "encoder_name": encoder_name,
        "feature_dim": feature_dim,
        "labels": sorted(EXPECTED_LABELS),
        "budget_tiers": sorted(EXPECTED_BUDGETS),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.pkl")
    try:
        joblib.dump(artifact, tmp_path, compress=3)
        tmp_path.replace(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    size_kb = output_path.stat().st_size / 1024
    log.info("Model artifact saved -> %s (%.1f KB)", output_path, size_kb)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(
        description="Train the ML Micro-Router (XGBoost) for budget-aware RAG routing."
    )
    p.add_argument("--csv", type=Path, default=Path("data/router_training_data.csv"))
    p.add_argument("--output", type=Path, default=Path("data/router_model.pkl"))
    p.add_argument(
        "--encoder",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name or local path.",
    )
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing model artifact without prompting.",
    )
    args = p.parse_args()
    return TrainConfig(
        csv_path=args.csv,
        model_output_path=args.output,
        encoder_name=args.encoder,
        n_estimators=args.n_estimators,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        seed=args.seed,
        force=args.force,
    )


# ── Entry Point ────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = _parse_args()

    log.info("=" * 60)
    log.info("Router Training Pipeline")
    log.info("  CSV        : %s", cfg.csv_path)
    log.info("  Output     : %s", cfg.model_output_path)
    log.info("  Encoder    : %s", cfg.encoder_name)
    log.info("  Estimators : %d | Test size: %.0f%%", cfg.n_estimators, cfg.test_size * 100)
    log.info("=" * 60)

    # Guard: don't overwrite a valid artifact unless --force is set
    if cfg.model_output_path.exists() and not cfg.force:
        log.info(
            "Model artifact already exists at '%s'. Use --force to retrain.",
            cfg.model_output_path,
        )
        sys.exit(0)

    # Step 1: Load + validate
    try:
        df = load_and_validate(cfg)
    except (FileNotFoundError, ValueError) as exc:
        log.error("Data validation failed: %s", exc)
        sys.exit(1)

    # Step 2: Load encoder
    log.info("Loading sentence encoder '%s' ...", cfg.encoder_name)
    try:
        encoder = SentenceTransformer(cfg.encoder_name)
    except Exception as exc:
        log.error("Failed to load encoder: %s", exc)
        sys.exit(1)

    # Step 3: Build features
    try:
        X, y = build_features(df, encoder)
    except Exception as exc:
        log.error("Feature engineering failed: %s", exc, exc_info=True)
        sys.exit(1)

    # Step 4: Train + evaluate
    try:
        clf = train_and_evaluate(X, y, cfg, df=df)
    except Exception as exc:
        log.error("Training failed: %s", exc, exc_info=True)
        sys.exit(1)

    # Step 5: Save artifact
    try:
        save_artifact(clf, cfg.encoder_name, X.shape[1], cfg.model_output_path)
    except Exception as exc:
        log.error("Failed to save model artifact: %s", exc, exc_info=True)
        sys.exit(1)

    log.info("=" * 60)
    log.info("Training complete.")
    log.info("Next step: streamlit run app.py  OR  python ablation_runner.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()