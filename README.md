# ⚖️ Legal Document Summarization with Intent-Based Evaluation

A research repository for automatic summarization of Indian legal case judgments, with a novel **intent-phrase-based evaluation framework** that correlates strongly (F1 Pearson r = 0.86) with human judgment.

> Originally conceived in 2021. Rebuilt in 2026 using modern NLP tooling.

---

## 📌 Overview

Indian Supreme Court judgments average **4,500+ words**. This project:

1. **Scrapes** judgments from [IndianKanoon](https://indiankanoon.org) across 4 crime categories
2. **Annotates** intent phrases — sentences that reveal the core legal intent of a case
3. **Fine-tunes** [InLegalBERT](https://huggingface.co/law-ai/InLegalBERT) for automatic intent phrase extraction (NER)
4. **Summarizes** documents using LetSum, BERT, Longformer (LED), and a Graph-based model
5. **Evaluates** summaries using custom Intent-based Precision/Recall/F1 + BERTScore

---

## 🗂️ Repository Structure

```
legal-summarization/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/               # Scraped judgment JSON files
│   ├── annotated/         # Intent-phrase annotations in BIO format
│   │   └── sample_annotations.json
│   └── processed/         # Tokenized / train-val-test splits
│
├── src/
│   ├── scraper.py         # IndianKanoon scraper
│   ├── annotator.py       # LLM-assisted pre-annotation helper
│   ├── intent_extractor.py# Fine-tune InLegalBERT for intent NER
│   ├── summarizer.py      # LetSum, BERT, LED, Graph summarizers
│   └── evaluate.py        # Intent-F1, ROUGE, BERTScore
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_annotation_analysis.ipynb
│   ├── 03_intent_extraction.ipynb
│   ├── 04_summarization.ipynb
│   └── 05_evaluation.ipynb
│
├── results/               # CSVs and plots from experiments
└── models/                # Saved checkpoints (see HuggingFace Hub)
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/legal-summarization.git
cd legal-summarization
pip install -r requirements.txt
```

### 2. Scrape Data

```bash
python src/scraper.py --category murder --max_docs 200 --output data/raw/
```

Categories: `murder`, `robbery`, `corruption`, `land_dispute`

### 3. LLM-Assisted Annotation

```bash
python src/annotator.py \
  --input data/raw/ \
  --output data/annotated/ \
  --api_key YOUR_ANTHROPIC_KEY   # or --use_ollama for local LLM
```

Then review annotations in [Label Studio](https://labelstud.io/) (free, local).

### 4. Train Intent Extractor

```bash
python src/intent_extractor.py \
  --data data/annotated/ \
  --model law-ai/InLegalBERT \
  --output models/intent_extractor/
```

### 5. Run Summarization + Evaluation

```bash
python src/summarizer.py --input data/raw/ --method graph --output results/
python src/evaluate.py --summaries results/ --annotations data/annotated/
```

---

## 📊 Results (Original 2021 Study)

| Summarization Method | Intent Precision | Intent Recall | Intent F1 |
|---|---|---|---|
| LetSum | 0.278 | 0.149 | 0.194 |
| Pre-trained BERT | 0.252 | 0.149 | 0.187 |
| **Graph-based (Ours)** | **0.282** | **0.159** | **0.203** |

**Human evaluation:** Intent-based F1 showed Pearson correlation of **0.8646** with human judgment.

---

## 🧪 Compute Requirements

All experiments designed to run on **free-tier hardware**:

| Resource | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| GPU | None (CPU fallback) | T4 (Colab/Kaggle free) |
| Storage | 1 GB | 5 GB |

Free GPU options: [Google Colab](https://colab.research.google.com) · [Kaggle Notebooks](https://www.kaggle.com/code) (30 hrs/week)

---

## 📚 Dataset

- **Source:** [IndianKanoon.org](https://indiankanoon.org)
- **Categories:** Murder · Robbery · Corruption · Land Dispute
- **Original size:** 93 documents (2021) → Target: 500+ documents

Intent phrases are annotated in **BIO format** (see `data/annotated/sample_annotations.json`).

---

## 🙏 Citation

If you use this work, please cite:

```bibtex
@misc{legalsumm2021,
  title     = {Intent-Based Evaluation of Legal Document Summarization},
  author    = {Nandy, Abhilash and Dam, Arpan and Halder, Tanurima},
  year      = {2021},
  note      = {IIT Kharagpur}
}
```

---

## 📄 License

MIT License. Data scraped from IndianKanoon is subject to their terms of use.
