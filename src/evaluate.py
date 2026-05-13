"""
src/evaluate.py
---------------
Evaluates generated summaries using three metric families:

  1. Intent-based metrics (novel, from original 2021 study)
       - Intent Precision: intent sentences in summary / total summary sentences
       - Intent Recall:    summary sentences with intent / total annotated intents
       - Intent F1

  2. ROUGE (standard)
       - ROUGE-1, ROUGE-2, ROUGE-L

  3. BERTScore (semantic similarity, 2026 standard)
       - Precision, Recall, F1 using DeBERTa

Usage:
    python src/evaluate.py \
        --summaries results/ \
        --annotations data/annotated/ \
        --output results/evaluation.csv
"""

import json
import logging
import argparse
from pathlib import Path

import re

import pandas as pd
import numpy as np


def sent_tokenize_safe(text: str) -> list[str]:
    """Sentence tokenizer with NLTK fallback to simple regex split."""
    try:
        import nltk
        for res in ["punkt", "punkt_tab"]:
            try:
                nltk.data.find(f"tokenizers/{res}")
                return nltk.sent_tokenize(text)
            except LookupError:
                try:
                    nltk.download(res, quiet=True)
                    return nltk.sent_tokenize(text)
                except Exception:
                    pass
    except Exception:
        pass
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent-based metrics (original contribution)
# ---------------------------------------------------------------------------

def intent_precision(summary: str, annotated_intents: list[str]) -> float:
    """
    Fraction of summary sentences that contain an intent phrase.

    Intent Precision = |summary sentences containing intent phrase|
                       ─────────────────────────────────────────
                       |total sentences in summary|
    """
    sentences = sent_tokenize_safe(summary)
    if not sentences:
        return 0.0

    intent_set = {p.lower().strip()[:60] for p in annotated_intents}
    hits = sum(
        1 for sent in sentences
        if any(phrase in sent.lower() for phrase in intent_set)
    )
    return hits / len(sentences)


def intent_recall(summary: str, annotated_intents: list[str]) -> float:
    """
    Fraction of annotated intent phrases covered by the summary.

    Intent Recall = |annotated intents appearing in summary|
                    ────────────────────────────────────────
                    |total annotated intent phrases|
    """
    if not annotated_intents:
        return 0.0

    summary_lower = summary.lower()
    covered = sum(
        1 for phrase in annotated_intents
        if phrase.lower().strip()[:60] in summary_lower
    )
    return covered / len(annotated_intents)


def intent_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# ROUGE
# ---------------------------------------------------------------------------

def compute_rouge(summary: str, reference: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L against a reference text."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, summary)
        return {
            "rouge1_f": round(scores["rouge1"].fmeasure, 4),
            "rouge2_f": round(scores["rouge2"].fmeasure, 4),
            "rougeL_f": round(scores["rougeL"].fmeasure, 4),
        }
    except ImportError:
        log.warning("rouge_score not installed. Run: pip install rouge-score")
        return {"rouge1_f": None, "rouge2_f": None, "rougeL_f": None}


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

_bertscore_fn = None

def compute_bertscore(summary: str, reference: str) -> dict:
    """Compute BERTScore F1 using DeBERTa-xlarge-mnli."""
    global _bertscore_fn
    try:
        import bert_score
        if _bertscore_fn is None:
            log.info("Loading BERTScore model (first time, ~1.4GB)...")
        P, R, F = bert_score.score(
            [summary], [reference],
            model_type="microsoft/deberta-xlarge-mnli",
            lang="en",
            verbose=False,
        )
        return {
            "bertscore_p": round(P[0].item(), 4),
            "bertscore_r": round(R[0].item(), 4),
            "bertscore_f": round(F[0].item(), 4),
        }
    except ImportError:
        log.warning("bert_score not installed. Run: pip install bert-score")
        return {"bertscore_p": None, "bertscore_r": None, "bertscore_f": None}


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(summaries_dir: Path, annotations_dir: Path, output_path: Path, skip_bertscore: bool):
    """
    Match summary files against annotation files and compute all metrics.
    Both directories are searched recursively; files are matched by doc_id.
    """
    # Index annotations by doc_id
    annotation_index = {}
    for ann_path in annotations_dir.glob("*.json"):
        ann = json.loads(ann_path.read_text())
        annotation_index[ann["doc_id"]] = ann

    log.info(f"Found {len(annotation_index)} annotation files.")

    rows = []

    # Collect all summary files across method subdirectories
    summary_files = list(summaries_dir.glob("**/*.json"))
    log.info(f"Found {len(summary_files)} summary files.")

    for path in tqdm(summary_files, desc="Evaluating"):
        summary_doc = json.loads(path.read_text())
        doc_id = summary_doc.get("doc_id")

        if doc_id not in annotation_index:
            log.debug(f"No annotation for {doc_id}, skipping.")
            continue

        ann = annotation_index[doc_id]
        summary_text = summary_doc.get("summary", "")
        original_text = ann.get("bio_sentences", [])
        full_text = " ".join(s["sentence"] for s in original_text)
        intents = ann.get("intent_phrases", [])

        # Intent metrics
        prec = intent_precision(summary_text, intents)
        rec  = intent_recall(summary_text, intents)
        f1   = intent_f1(prec, rec)

        row = {
            "doc_id":           doc_id,
            "category":         summary_doc.get("category"),
            "method":           summary_doc.get("method"),
            "summary_words":    summary_doc.get("summary_words"),
            "original_words":   summary_doc.get("original_words"),
            "compression_ratio": round(
                summary_doc.get("summary_words", 0) /
                max(summary_doc.get("original_words", 1), 1), 4
            ),
            "intent_precision": round(prec, 4),
            "intent_recall":    round(rec, 4),
            "intent_f1":        round(f1, 4),
        }

        # ROUGE
        rouge = compute_rouge(summary_text, full_text)
        row.update(rouge)

        # BERTScore (slow; skip if requested)
        if not skip_bertscore:
            bs = compute_bertscore(summary_text, full_text)
            row.update(bs)

        rows.append(row)

    if not rows:
        log.error("No matched (summary, annotation) pairs found. Check doc_ids match.")
        return

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info(f"Results saved to {output_path}")

    # Print aggregate summary
    print_summary(df)


def print_summary(df: pd.DataFrame):
    """Print a formatted summary table grouped by method."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS — Mean across all documents")
    print("=" * 70)

    metric_cols = [c for c in [
        "intent_precision", "intent_recall", "intent_f1",
        "rouge1_f", "rouge2_f", "rougeL_f",
        "bertscore_f",
    ] if c in df.columns]

    grouped = df.groupby("method")[metric_cols].mean().round(4)
    print(grouped.to_string())

    print("\n" + "=" * 70)
    print("Results by Category x Method")
    print("=" * 70)
    if "category" in df.columns:
        cat_grouped = df.groupby(["method", "category"])["intent_f1"].mean().round(4)
        print(cat_grouped.to_string())


# ---------------------------------------------------------------------------
# Pearson correlation with human scores (from original study)
# ---------------------------------------------------------------------------

def correlate_with_human(eval_csv: Path, human_scores_csv: Path):
    """
    Compute Pearson correlation between automatic metrics and human scores.
    human_scores_csv must have columns: doc_id, method, human_score (0 or 1)
    """
    from scipy import stats

    df_auto   = pd.read_csv(eval_csv)
    df_human  = pd.read_csv(human_scores_csv)
    df_merged = df_auto.merge(df_human, on=["doc_id", "method"])

    metric_cols = ["intent_precision", "intent_recall", "intent_f1", "rouge1_f", "rougeL_f"]

    print("\n" + "=" * 50)
    print("Pearson Correlation with Human Scores")
    print("=" * 50)
    for col in metric_cols:
        if col in df_merged.columns:
            r, p = stats.pearsonr(df_merged[col].dropna(), df_merged["human_score"].dropna())
            print(f"  {col:30s}: r = {r:.4f}  (p = {p:.4f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate legal document summaries")
    parser.add_argument("--summaries",   type=Path, required=True,
                        help="Directory of summary JSONs (may have method subdirs)")
    parser.add_argument("--annotations", type=Path, required=True,
                        help="Directory of annotated BIO JSONs")
    parser.add_argument("--output",      type=Path, default=Path("results/evaluation.csv"))
    parser.add_argument("--skip_bertscore", action="store_true",
                        help="Skip BERTScore (faster, useful for quick runs)")
    parser.add_argument("--human_scores", type=Path, default=None,
                        help="CSV with human evaluation scores (optional)")

    args = parser.parse_args()

    evaluate(args.summaries, args.annotations, args.output, args.skip_bertscore)

    if args.human_scores and args.human_scores.exists():
        correlate_with_human(args.output, args.human_scores)


if __name__ == "__main__":
    main()
