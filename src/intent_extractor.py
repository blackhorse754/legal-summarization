"""
src/intent_extractor.py
-----------------------
Fine-tunes InLegalBERT (law-ai/InLegalBERT) as a sentence-level intent
phrase classifier (binary: INTENT vs O) on BIO-annotated data.

Training runs comfortably on:
  - Google Colab free (T4 GPU)
  - Kaggle free (T4/P100)
  - CPU-only (slow but works for small datasets)

Usage:
    # Train
    python src/intent_extractor.py train \
        --data data/annotated/ \
        --output models/intent_extractor/

    # Predict on a new judgment
    python src/intent_extractor.py predict \
        --model models/intent_extractor/ \
        --input data/raw/murder/123456.json
"""

import json
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BASE_MODEL = "law-ai/InLegalBERT"
LABEL2ID  = {"O": 0, "B-INTENT": 1}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}
MAX_LEN   = 256


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IntentDataset(Dataset):
    def __init__(self, sentences: list[str], labels: list[int], tokenizer):
        self.encodings = tokenizer(
            sentences,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


def load_annotated_data(data_dir: Path) -> tuple[list[str], list[int]]:
    """Load all BIO-annotated files and return (sentences, labels) lists."""
    sentences, labels = [], []
    files = list(data_dir.glob("*.json"))
    log.info(f"Loading {len(files)} annotated files from {data_dir}")

    for path in files:
        doc = json.loads(path.read_text())
        for item in doc.get("bio_sentences", []):
            sentences.append(item["sentence"])
            labels.append(LABEL2ID.get(item["label"], 0))

    log.info(f"Total sentences: {len(sentences)} | Intent: {sum(labels)} | O: {len(labels)-sum(labels)}")
    return sentences, labels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    # Focus on F1 for the INTENT class
    from sklearn.metrics import f1_score, precision_score, recall_score
    return {
        "f1_intent":        f1_score(labels, preds, pos_label=1, zero_division=0),
        "precision_intent": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall_intent":    recall_score(labels, preds, pos_label=1, zero_division=0),
        "accuracy":         (preds == labels).mean(),
    }


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(data_dir: Path, output_dir: Path, epochs: int, batch_size: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences, labels = load_annotated_data(data_dir)

    if len(sentences) < 20:
        log.error("Too few samples. Annotate more documents first.")
        return

    # Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        sentences, labels, test_size=0.15, stratify=labels, random_state=42
    )

    log.info(f"Train: {len(X_train)} | Val: {len(X_val)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds = IntentDataset(X_train, y_train, tokenizer)
    val_ds   = IntentDataset(X_val,   y_val,   tokenizer)

    # Class weights to handle imbalance (intent phrases are sparse)
    neg, pos = len(y_train) - sum(y_train), sum(y_train)
    pos_weight = neg / max(pos, 1)
    log.info(f"Class balance — O: {neg} | INTENT: {pos} | pos_weight: {pos_weight:.2f}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_intent",
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="none",     # set to "wandb" to enable experiment tracking
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # Save model + tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info(f"Model saved to {output_dir}")

    # Full classification report
    preds_output = trainer.predict(val_ds)
    preds = np.argmax(preds_output.predictions, axis=-1)
    print("\n" + classification_report(y_val, preds, target_names=["O", "B-INTENT"]))


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def predict(model_dir: Path, input_path: Path) -> list[dict]:
    """
    Run intent extraction on a raw judgment JSON.
    Returns list of {sentence, label, confidence}.
    """
    from transformers import pipeline

    log.info(f"Loading model from {model_dir}")
    clf = pipeline(
        "text-classification",
        model=str(model_dir),
        tokenizer=str(model_dir),
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=MAX_LEN,
    )

    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    judgment = json.loads(input_path.read_text())
    sentences = nltk.sent_tokenize(judgment["full_text"])

    results = []
    batch_size = 32
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        outputs = clf(batch)
        for sent, out in zip(batch, outputs):
            results.append({
                "sentence":   sent,
                "label":      out["label"],
                "confidence": round(out["score"], 4),
            })

    intent_sentences = [r for r in results if r["label"] == "B-INTENT"]
    log.info(f"Found {len(intent_sentences)} intent phrases out of {len(sentences)} sentences.")

    for r in intent_sentences:
        print(f"  [{r['confidence']:.2f}] {r['sentence'][:100]}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Intent phrase extractor (InLegalBERT)")
    sub = parser.add_subparsers(dest="command", required=True)

    # Train
    train_p = sub.add_parser("train")
    train_p.add_argument("--data",   type=Path, required=True)
    train_p.add_argument("--output", type=Path, default=Path("models/intent_extractor"))
    train_p.add_argument("--epochs", type=int, default=5)
    train_p.add_argument("--batch_size", type=int, default=16)

    # Predict
    pred_p = sub.add_parser("predict")
    pred_p.add_argument("--model", type=Path, required=True)
    pred_p.add_argument("--input", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "train":
        train(args.data, args.output, args.epochs, args.batch_size)
    elif args.command == "predict":
        predict(args.model, args.input)


if __name__ == "__main__":
    main()
