"""
src/annotator.py
----------------
LLM-assisted pre-annotation of intent phrases in legal judgments.

Supports:
  - Anthropic Claude API  (--api_key / ANTHROPIC_API_KEY env var)
  - Local Ollama models   (--use_ollama --ollama_model mistral)

The output is a JSON file per document in BIO-tagged sentence format,
ready for human review in Label Studio.

Usage:
    # Cloud (Claude)
    python src/annotator.py --input data/raw/murder/ --output data/annotated/ \
        --api_key sk-ant-...

    # Local (Ollama — free, no API key)
    python src/annotator.py --input data/raw/murder/ --output data/annotated/ \
        --use_ollama --ollama_model mistral
"""

import json
import logging
import argparse
import os
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a legal NLP annotation expert specialising in Indian court judgments.
Your task is to identify "intent phrases" — sentences or clauses that directly reveal the
criminal or legal intent at the heart of the case (e.g. the act of murder, robbery, corruption,
or land encroachment).

Rules:
1. Return ONLY a valid JSON object — no preamble, no markdown fences.
2. Format: {"intent_phrases": ["sentence 1", "sentence 2", ...], "primary_intent": "<label>"}
3. primary_intent must be one of: murder, robbery, corruption, land_dispute, other
4. Limit to the 3-7 most important intent phrases.
5. Copy phrases verbatim from the input text.
"""

USER_TEMPLATE = """Document category: {category}

Judgment text (first 3000 words):
\"\"\"
{text}
\"\"\"

Identify the intent phrases."""


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def call_claude(text: str, category: str, api_key: str) -> dict:
    """Call Anthropic Claude API."""
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": USER_TEMPLATE.format(
                category=category,
                text=text[:12000],   # ~3000 words
            )}
        ],
    }
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    raw = resp.json()["content"][0]["text"]
    return json.loads(raw)


def call_ollama(text: str, category: str, model: str, base_url: str) -> dict:
    """Call a local Ollama model."""
    prompt = SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE.format(
        category=category, text=text[:12000]
    )
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=120)
    resp.raise_for_status()
    raw = resp.json()["response"]

    # Strip possible markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ---------------------------------------------------------------------------
# BIO conversion
# ---------------------------------------------------------------------------

def to_bio_format(judgment: dict, annotation: dict) -> dict:
    """
    Convert intent phrase annotations to BIO-tagged sentence list.
    Each sentence is tagged as B-INTENT, I-INTENT, or O.
    """
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    sentences = nltk.sent_tokenize(judgment["full_text"])
    intent_phrases = set(p.lower().strip() for p in annotation.get("intent_phrases", []))

    bio_sentences = []
    for sent in sentences:
        # Simple heuristic: mark sentence as INTENT if it contains an intent phrase
        label = "O"
        for phrase in intent_phrases:
            if phrase[:60] in sent.lower():   # match first 60 chars to handle truncation
                label = "B-INTENT"
                break
        bio_sentences.append({"sentence": sent, "label": label})

    return {
        "doc_id": judgment["doc_id"],
        "title": judgment["title"],
        "category": judgment["category"],
        "primary_intent": annotation.get("primary_intent", judgment["category"]),
        "intent_phrases": annotation.get("intent_phrases", []),
        "bio_sentences": bio_sentences,
        "num_sentences": len(bio_sentences),
        "num_intent_sentences": sum(1 for s in bio_sentences if s["label"] != "O"),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def annotate(
    input_dir: Path,
    output_dir: Path,
    api_key: str | None,
    use_ollama: bool,
    ollama_model: str,
    ollama_base_url: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_files = list(input_dir.glob("**/*.json"))

    if not json_files:
        log.error(f"No JSON files found in {input_dir}")
        return

    log.info(f"Annotating {len(json_files)} documents from {input_dir}")
    success, failed = 0, 0

    for path in tqdm(json_files, desc="Annotating"):
        out_path = output_dir / path.name
        if out_path.exists():
            log.debug(f"Skipping already-annotated {path.name}")
            continue

        judgment = json.loads(path.read_text())

        try:
            if use_ollama:
                annotation = call_ollama(
                    judgment["full_text"],
                    judgment["category"],
                    ollama_model,
                    ollama_base_url,
                )
            else:
                annotation = call_claude(
                    judgment["full_text"],
                    judgment["category"],
                    api_key,
                )

            result = to_bio_format(judgment, annotation)
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            success += 1

        except Exception as e:
            log.warning(f"Failed on {path.name}: {e}")
            failed += 1

    log.info(f"Done. Annotated: {success}, Failed: {failed}")
    log.info(
        "\nNEXT STEP: Review annotations in Label Studio.\n"
        "  1. pip install label-studio\n"
        "  2. label-studio start\n"
        "  3. Import files from: " + str(output_dir)
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM-assisted intent phrase annotation")
    parser.add_argument("--input",  type=Path, required=True, help="Directory of raw judgment JSONs")
    parser.add_argument("--output", type=Path, default=Path("data/annotated"))

    # Claude
    parser.add_argument("--api_key", default=os.environ.get("ANTHROPIC_API_KEY"),
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")

    # Ollama (local, free)
    parser.add_argument("--use_ollama", action="store_true", help="Use local Ollama instead of Claude")
    parser.add_argument("--ollama_model", default="mistral", help="Ollama model name")
    parser.add_argument("--ollama_base_url", default="http://localhost:11434")

    args = parser.parse_args()

    if not args.use_ollama and not args.api_key:
        parser.error("Provide --api_key or set ANTHROPIC_API_KEY, or use --use_ollama")

    annotate(
        input_dir=args.input,
        output_dir=args.output,
        api_key=args.api_key,
        use_ollama=args.use_ollama,
        ollama_model=args.ollama_model,
        ollama_base_url=args.ollama_base_url,
    )


if __name__ == "__main__":
    main()
