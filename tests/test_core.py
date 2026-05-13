"""
tests/test_core.py
------------------
Unit tests for summarization and evaluation functions.
Runs without GPU, no network calls.
"""

import sys
sys.path.insert(0, '.')

import pytest
from src.summarizer import graph_summarize, letsum_summarize
from src.evaluate import intent_precision, intent_recall, intent_f1


SAMPLE_TEXT = """
The appellant sole accused is convicted of committing the murder of one Shailesh Balkrishna Junghare.
He was assaulted with the help of a knife on 23/09/2014 at about 7:15 p.m.
The learned Sessions Judge, Nagpur convicted the appellant for the offence under Section 302 of IPC.
The prosecution examined thirteen witnesses in all to prove their case.
The court carefully examined the chain of evidence presented by both sides.
Medical evidence confirmed death due to stab injuries.
The defence argued that the accused was not present at the scene of crime.
We find no merit in the appeal and the conviction is accordingly upheld.
The sentence of life imprisonment awarded by the trial court is confirmed.
"""

INTENT_PHRASES = [
    "The appellant sole accused is convicted of committing the murder of one Shailesh Balkrishna Junghare.",
    "convicted the appellant for the offence under Section 302 of IPC",
]


class TestSummarizers:
    def test_graph_returns_string(self):
        result = graph_summarize(SAMPLE_TEXT, ratio=0.3)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_graph_shorter_than_original(self):
        result = graph_summarize(SAMPLE_TEXT, ratio=0.3)
        assert len(result.split()) < len(SAMPLE_TEXT.split())

    def test_letsum_returns_string(self):
        result = letsum_summarize(SAMPLE_TEXT, ratio=0.3)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_letsum_ratio(self):
        full_sentences = 9  # roughly
        result = letsum_summarize(SAMPLE_TEXT, ratio=0.5)
        # Should have fewer sentences than original
        from src.summarizer import sent_tokenize_safe
        n = len(sent_tokenize_safe(result))
        assert n <= full_sentences

    def test_short_text_passthrough(self):
        short = "This is a very short text."
        result = graph_summarize(short, ratio=0.3)
        assert isinstance(result, str)


class TestEvaluationMetrics:
    def test_intent_precision_full_match(self):
        # Summary IS the intent phrase
        summary = INTENT_PHRASES[0]
        prec = intent_precision(summary, INTENT_PHRASES)
        assert prec > 0.0

    def test_intent_precision_no_match(self):
        summary = "The court examined the evidence carefully."
        prec = intent_precision(summary, INTENT_PHRASES)
        assert prec == 0.0

    def test_intent_recall_full_coverage(self):
        # Summary contains all intent phrases
        summary = " ".join(INTENT_PHRASES)
        rec = intent_recall(summary, INTENT_PHRASES)
        assert rec == 1.0

    def test_intent_recall_no_coverage(self):
        summary = "The court examined witnesses."
        rec = intent_recall(summary, INTENT_PHRASES)
        assert rec == 0.0

    def test_f1_zero_when_both_zero(self):
        assert intent_f1(0.0, 0.0) == 0.0

    def test_f1_harmonic_mean(self):
        p, r = 0.5, 0.5
        f1 = intent_f1(p, r)
        assert abs(f1 - 0.5) < 1e-6

    def test_f1_asymmetric(self):
        p, r = 1.0, 0.5
        f1 = intent_f1(p, r)
        assert abs(f1 - (2 * 1.0 * 0.5 / 1.5)) < 1e-6

    def test_empty_intents(self):
        rec = intent_recall("some summary", [])
        assert rec == 0.0
