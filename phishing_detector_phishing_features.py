"""
Moduł ekstrakcji cech i budowy potoku dla lokalnego filtra antyphishingowego (Opcja A).
Współdzielony pomiędzy notatnikiem `phishing_detector.ipynb` a skryptem `predict.py`,
co gwarantuje bezbłędną serializację i deserializację modelu (.joblib) bez zależności od __main__.
"""

from __future__ import annotations

import math
import re
from urllib.parse import urlsplit
from collections import Counter
from typing import Iterable, List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler

# Ogranicza koszt dalszej ekstrakcji do maksymalnie 4096 znaków na próbkę.
MAX_TEXT_LEN_FAST = 4_096

_IPV4_RE = re.compile(r"(?:https?://)?(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:/|$)")
_PWD_INPUT_RE = re.compile(r"<input[^>]+type\s*=\s*['\"]?password", re.IGNORECASE)
_JS_OBF_RE = re.compile(
    r"(?:eval\s*\(|atob\s*\(|fromcharcode|document\.write\s*\()",
    re.IGNORECASE,
)
_URGENCY_RE = re.compile(
    r"\b(?:verify|suspend(?:ed)?|urgent|account|password|login|sign[\s-]?in|"
    r"confirm|security|update|bank|wallet|expired|billing|invoice)\b",
    re.IGNORECASE,
)
_HTML_TAG_RE = re.compile(r"(?:<!doctype|<html|<head|<body|<form|<script)", re.IGNORECASE)
_DOMAIN_TERMS = ("login", "verify", "secure", "account", "update", "wallet", "signin", "password", "bank")


def fast_truncate(texts: Iterable[str], max_len: int = MAX_TEXT_LEN_FAST) -> List[str]:
    """Przycina tekst do ustalonego prefiksu; dalszy koszt ogranicza max_len."""
    return [t[:max_len] if isinstance(t, str) else "" for t in texts]


def _shannon_entropy_fast(sample: str) -> float:
    """Oblicza entropię Shannona na krótkim prefiksie (wykrywanie domen DGA i ciągów Base64)."""
    if not sample:
        return 0.0
    counts = Counter(sample)
    inv_len = 1.0 / len(sample)
    return -sum((c * inv_len) * math.log2(c * inv_len) for c in counts.values())


class FastLexicalExtractor(BaseEstimator, TransformerMixin):
    """
    Lekki ekstraktor 20 cech leksykalnych i strukturalnych (float32).
    Operuje wyłącznie na krótkich prefiksach ciągu wejściowego (URL / HTML / SMS).
    """

    FEATURE_NAMES = [
        "log_len",
        "is_url_only",
        "is_html",
        "digit_ratio",
        "special_char_ratio",
        "shannon_entropy",
        "has_ipv4",
        "has_punycode",
        "at_symbol_count",
        "url_depth_slashes",
        "subdomain_dots",
        "has_password_input",
        "has_js_obfuscation",
        "urgency_keyword_hits",
        "host_length",
        "host_digit_ratio",
        "subdomain_count",
        "suspicious_domain_term",
        "suspicious_path_term",
        "hyphen_count_host",
    ]

    def __init__(self, max_len: int = MAX_TEXT_LEN_FAST):
        self.max_len = max_len

    def fit(self, X: Iterable[str], y=None):
        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:
        rows = []
        for raw in X:
            s = raw[: self.max_len] if isinstance(raw, str) else ""
            n = len(s)
            if n == 0:
                rows.append([0.0] * len(self.FEATURE_NAMES))
                continue

            head_120 = s[:120]
            head_256 = s[:256]
            lower_head_256 = head_256.lower()
            lower_s = s.lower()

            log_len = math.log1p(n)
            stripped = s.strip()
            is_url_only = float(
                stripped.startswith(("http://", "https://", "www."))
                and ("\n" not in stripped)
                and (" " not in stripped)
            )
            is_html = float(bool(_HTML_TAG_RE.search(head_256)))

            digits = sum(ch.isdigit() for ch in head_256)
            specials = sum((not ch.isalnum()) and (not ch.isspace()) for ch in head_256)
            inv_256 = 1.0 / len(head_256)

            digit_ratio = digits * inv_256
            special_char_ratio = specials * inv_256
            entropy = _shannon_entropy_fast(head_256)

            has_ipv4 = float(bool(_IPV4_RE.search(head_256)))
            has_punycode = float("xn--" in lower_head_256)
            at_count = float(head_256.count("@"))
            slash_count = float(head_256.count("/"))
            dot_count = float(head_120.count("."))

            has_pwd = float(bool(_PWD_INPUT_RE.search(lower_s)))
            has_js_obf = float(bool(_JS_OBF_RE.search(lower_s)))
            urgency_hits = float(min(len(_URGENCY_RE.findall(s[:1500])), 12))

            candidate = stripped.split()[0].strip("<>()[]{}\"'.,;") if stripped.split() else ""
            if candidate.lower().startswith("www."):
                candidate = "http://" + candidate
            try:
                parsed = urlsplit(candidate if "://" in candidate else "//" + candidate)
                host = (parsed.hostname or "").lower().rstrip(".")[:253]
                path_query = (parsed.path + "?" + parsed.query).lower()[:512]
            except ValueError:  # malformed bracketed/IPv6-like input
                host, path_query = "", ""
            labels = [part for part in host.split(".") if part]
            host_digits = sum(ch.isdigit() for ch in host)
            host_length = min(len(host), 253)
            domain_term = float(any(term in host for term in _DOMAIN_TERMS))
            path_term = float(any(term in path_query for term in _DOMAIN_TERMS))

            rows.append(
                [
                    log_len,
                    is_url_only,
                    is_html,
                    digit_ratio,
                    special_char_ratio,
                    entropy,
                    has_ipv4,
                    has_punycode,
                    at_count,
                    slash_count,
                    dot_count,
                    has_pwd,
                    has_js_obf,
                    urgency_hits,
                    math.log1p(host_length),
                    host_digits / max(host_length, 1),
                    float(min(max(len(labels) - 2, 0), 10)),
                    domain_term,
                    path_term,
                    float(min(host.count("-"), 10)),
                ]
            )

        return np.asarray(rows, dtype=np.float32)


def build_option_a_pipeline(
    n_features_word: int = 2**15,
    n_features_char: int = 2**15,
    alpha: float = 1e-5,
    random_state: int = 42,
) -> Pipeline:
    """
    Buduje bezstanowy, ultralekki potok Opcji A (Ultra-Fast & Ultra-Compact):
    1. Przycięcie wejścia do 4096 znaków (fast_truncate).
    2. Równoległa wektoryzacja haszująca (słowna 1-2 + znakowa char_wb 3-4) + 20 cech strukturalnych.
    3. Liniowy klasyfikator logistyczny SGDClassifier(loss='log_loss').
    """
    features = FeatureUnion(
        transformer_list=[
            (
                "word_hash",
                HashingVectorizer(
                    n_features=n_features_word,
                    ngram_range=(1, 2),
                    token_pattern=r"(?u)\b\w\w+\b",
                    alternate_sign=False,
                    norm="l2",
                    dtype=np.float32,
                ),
            ),
            (
                "char_hash",
                HashingVectorizer(
                    n_features=n_features_char,
                    analyzer="char_wb",
                    ngram_range=(3, 4),
                    alternate_sign=False,
                    norm="l2",
                    dtype=np.float32,
                ),
            ),
            (
                "struct",
                Pipeline(
                    [
                        ("extract", FastLexicalExtractor(max_len=MAX_TEXT_LEN_FAST)),
                        ("scale", MaxAbsScaler()),
                    ]
                ),
            ),
        ]
    )

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=100,
        tol=1e-3,
        class_weight="balanced",
        random_state=random_state,
    )

    return Pipeline(
        [
            ("truncate", FunctionTransformer(fast_truncate, validate=False)),
            ("features", features),
            ("clf", clf),
        ]
    )
