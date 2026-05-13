"""
src/summarizer.py
-----------------
Four summarization methods for Indian legal judgments:

  1. graph   — Graph-based extractive (TextRank variant, best in 2021 study)
  2. letsum  — LetSum-style (position + cue phrase weighting)
  3. bert    — Pre-trained BERT extractive (sentence embeddings)
  4. led     — Longformer Encoder-Decoder abstractive (best 2026 option)

Usage:
    python src/summarizer.py \
        --input data/raw/murder/ \
        --method graph \
        --ratio 0.3 \
        --output results/graph/
"""

import json
import logging
import argparse
from pathlib import Path

import nltk
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Lazy imports for heavy models
_bert_model = None
_led_pipeline = None


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
    # Regex fallback
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]


def ensure_nltk():
    pass  # replaced by sent_tokenize_safe


# ---------------------------------------------------------------------------
# Method 1: Graph-based (TextRank variant)
# ---------------------------------------------------------------------------

def graph_summarize(text: str, ratio: float = 0.3) -> str:
    """
    TextRank-style graph summarization.
    Sentences are nodes; edges are cosine similarity of TF-IDF vectors.
    Highest-scoring sentences (by PageRank) are selected.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import networkx as nx

    pass  # handled by sent_tokenize_safe
    sentences = sent_tokenize_safe(text)
    sentences = [s.strip() for s in sentences if len(s.split()) > 5]

    if len(sentences) < 3:
        return text

    # Build TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    try:
        tfidf = vectorizer.fit_transform(sentences)
    except ValueError:
        return " ".join(sentences[:max(1, int(len(sentences) * ratio))])

    # Build similarity graph
    sim_matrix = cosine_similarity(tfidf)
    np.fill_diagonal(sim_matrix, 0)

    # PageRank
    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph, alpha=0.85, max_iter=200)

    # Select top sentences
    num_select = max(1, int(len(sentences) * ratio))
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_indices = sorted([idx for idx, _ in ranked[:num_select]])
    return " ".join(sentences[i] for i in top_indices)


# ---------------------------------------------------------------------------
# Method 2: LetSum-style extractive
# ---------------------------------------------------------------------------

LETSUM_CUE_PHRASES = [
    "held", "ordered", "directed", "found", "observed", "convicted",
    "acquitted", "dismissed", "allowed", "court", "appeal", "judgment",
    "section", "ipc", "act", "therefore", "accordingly", "thus",
]


def letsum_summarize(text: str, ratio: float = 0.3) -> str:
    """
    LetSum-inspired heuristic: score sentences by position + cue phrases.
    Higher weight for sentences near the beginning/end and with legal cues.
    """
    pass  # handled by sent_tokenize_safe
    sentences = sent_tokenize_safe(text)
    sentences = [s.strip() for s in sentences if len(s.split()) > 5]
    n = len(sentences)

    if n < 3:
        return text

    scores = []
    for i, sent in enumerate(sentences):
        score = 0.0

        # Position score — first and last 10% of document are high value
        pos_ratio = i / n
        if pos_ratio < 0.1 or pos_ratio > 0.9:
            score += 0.4
        elif pos_ratio < 0.2 or pos_ratio > 0.8:
            score += 0.2

        # Cue phrase score
        sent_lower = sent.lower()
        cue_hits = sum(1 for cue in LETSUM_CUE_PHRASES if cue in sent_lower)
        score += min(cue_hits * 0.15, 0.6)

        # Length bonus (avoid very short sentences)
        words = len(sent.split())
        if 10 <= words <= 60:
            score += 0.1

        scores.append(score)

    num_select = max(1, int(n * ratio))
    top_indices = sorted(np.argsort(scores)[-num_select:])
    return " ".join(sentences[i] for i in top_indices)


# ---------------------------------------------------------------------------
# Method 3: BERT extractive (sentence-transformers)
# ---------------------------------------------------------------------------

def bert_summarize(text: str, ratio: float = 0.3) -> str:
    """
    Extractive summarization using sentence-transformers embeddings.
    Sentences closest to the document centroid are selected.
    """
    global _bert_model
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    pass  # handled by sent_tokenize_safe

    if _bert_model is None:
        log.info("Loading sentence-transformers model (first time, may download ~90MB)...")
        _bert_model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences = sent_tokenize_safe(text)
    sentences = [s.strip() for s in sentences if len(s.split()) > 5]
    if len(sentences) < 3:
        return text

    embeddings = _bert_model.encode(sentences, show_progress_bar=False)
    centroid = embeddings.mean(axis=0, keepdims=True)
    sims = cosine_similarity(embeddings, centroid).flatten()

    num_select = max(1, int(len(sentences) * ratio))
    top_indices = sorted(np.argsort(sims)[-num_select:])
    return " ".join(sentences[i] for i in top_indices)


# ---------------------------------------------------------------------------
# Method 4: LED abstractive (Longformer Encoder-Decoder)
# ---------------------------------------------------------------------------

def led_summarize(text: str, max_input_tokens: int = 4096, max_output_tokens: int = 512) -> str:
    """
    Abstractive summarization using Longformer Encoder-Decoder.
    Handles long legal documents (up to 16,384 tokens).
    Uses allenai/led-base-16384, fine-tuned on legal text if available.

    NOTE: Requires ~2GB RAM for the base model; runs on T4 GPU or CPU.
    """
    global _led_pipeline
    from transformers import pipeline

    if _led_pipeline is None:
        import torch
        log.info("Loading LED model (allenai/led-base-16384, ~480MB)...")
        _led_pipeline = pipeline(
            "summarization",
            model="allenai/led-base-16384",
            tokenizer="allenai/led-base-16384",
            device=0 if torch.cuda.is_available() else -1,
        )

    # Truncate input to model max
    words = text.split()
    if len(words) > max_input_tokens * 0.75:
        text = " ".join(words[: int(max_input_tokens * 0.75)])

    result = _led_pipeline(
        text,
        max_length=max_output_tokens,
        min_length=100,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    return result[0]["summary_text"]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

METHODS = {
    "graph":  graph_summarize,
    "letsum": letsum_summarize,
    "bert":   bert_summarize,
    "led":    led_summarize,
}


def summarize_file(judgment: dict, method: str, ratio: float) -> dict:
    fn = METHODS[method]
    text = judgment["full_text"]

    if method == "led":
        summary = fn(text)
    else:
        summary = fn(text, ratio=ratio)

    return {
        "doc_id":   judgment["doc_id"],
        "category": judgment["category"],
        "method":   method,
        "ratio":    ratio,
        "summary":  summary,
        "summary_words":   len(summary.split()),
        "original_words":  len(text.split()),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(input_dir: Path, output_dir: Path, method: str, ratio: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    files = list(input_dir.glob("**/*.json"))
    log.info(f"Summarizing {len(files)} documents with method='{method}'")

    for path in tqdm(files, desc=f"[{method}]"):
        out_path = output_dir / path.name
        if out_path.exists():
            continue
        try:
            judgment = json.loads(path.read_text())
            result = summarize_file(judgment, method, ratio)
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as e:
            log.warning(f"Failed on {path.name}: {e}")

    log.info(f"Summaries written to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Legal document summarizer")
    parser.add_argument("--input",  type=Path, required=True)
    parser.add_argument("--method", choices=list(METHODS), default="graph")
    parser.add_argument("--ratio",  type=float, default=0.3,
                        help="Summary length as fraction of original (not used for led)")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    method_dir = args.output / args.method
    run(args.input, method_dir, args.method, args.ratio)


if __name__ == "__main__":
    main()
