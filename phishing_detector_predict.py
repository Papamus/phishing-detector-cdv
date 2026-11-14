"""
Samodzielny, lokalny skrypt predykcyjny i benchmarkowy dla modelu wytrenowanego
w notatniku `phishing_detector.ipynb` (Opcja A: Ultra-Fast & Ultra-Compact).
Nie korzysta z żadnych zewnętrznych API modeli — działa w 100% lokalnie na CPU.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Wymagane do deserializacji potoku scikit-learn zapisanego w phishing_detector.ipynb
import phishing_features  # noqa: F401

DATASET_URL = (
    "https://huggingface.co/datasets/ealvaradob/phishing-dataset/resolve/main/combined_reduced.json"
)


class LocalPhishingDetector:
    """Lokalny klasyfikator antyphishingowy Opcji A."""

    def __init__(
        self,
        model_path: Path = Path("models/phishing_detector_option_a.joblib"),
        threshold: Optional[float] = None,
    ):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Nie znaleziono pliku modelu: {model_path}. "
                "Uruchom notatnik `phishing_detector.ipynb`, aby zapisać model."
            )
        self.model_path = model_path
        artifact = joblib.load(model_path)
        self.pipeline = artifact["pipeline"]
        self.threshold = float(threshold) if threshold is not None else float(artifact["threshold"])
        self.dataset_source = artifact.get("dataset_source", "unknown")

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)[:, 1]

    def predict(self, texts: List[str]) -> np.ndarray:
        return (self.predict_proba(texts) >= self.threshold).astype(int)

    def inspect_single(self, text: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        prob = float(self.predict_proba([text])[0])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        pred = int(prob >= self.threshold)
        return {
            "prediction": pred,
            "label": "PHISHING" if pred == 1 else "BENIGN",
            "phishing_probability": round(prob, 6),
            "threshold": round(self.threshold, 4),
            "inference_time_ms": round(elapsed_ms, 4),
        }


def evaluate_samples(
    detector: LocalPhishingDetector,
    texts: List[str],
    labels: Optional[List[int]] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if texts:
        _ = detector.predict_proba(texts[: min(5, len(texts))])

    t0 = time.perf_counter()
    probas = detector.predict_proba(texts)
    preds = (probas >= detector.threshold).astype(int)
    elapsed_s = time.perf_counter() - t0

    model_kb = detector.model_path.stat().st_size / 1024.0
    scripts_kb = sum(
        Path(p).stat().st_size for p in ["phishing_features.py", "predict.py"] if Path(p).exists()
    ) / 1024.0

    summary: Dict[str, Any] = {
        "option": "A_ultra_fast",
        "trained_on": detector.dataset_source,
        "threshold": round(detector.threshold, 4),
        "num_samples": len(texts),
        "model_size_kb": round(model_kb, 2),
        "scripts_size_kb": round(scripts_kb, 2),
        "total_inference_sec": round(elapsed_s, 4),
        "ms_per_sample": round((elapsed_s / max(len(texts), 1)) * 1000.0, 4),
        "samples_per_sec": round(len(texts) / max(elapsed_s, 1e-9), 1),
    }

    if labels is not None:
        y_true = np.asarray(labels, dtype=int)
        summary["recall_phishing"] = round(float(recall_score(y_true, preds, zero_division=0)), 4)
        summary["precision_phishing"] = round(float(precision_score(y_true, preds, zero_division=0)), 4)
        summary["f1_phishing"] = round(float(f1_score(y_true, preds, zero_division=0)), 4)
        summary["confusion_matrix"] = confusion_matrix(y_true, preds).tolist()

    print("\n" + "=" * 64)
    print("EWALUACJA LOKALNEGO FILTRA ANTYPHISHINGOWEGO (OPCJA A)")
    print("=" * 64)
    print(f"Źródło treningu     : {detector.dataset_source}")
    print(f"1. Wielkość modelu  : {summary['model_size_kb']} KB (skrypty runtime: {summary['scripts_size_kb']} KB)")
    print(f"2. Szybkość         : {summary['total_inference_sec']} s dla {len(texts):,} próbek "
          f"({summary['ms_per_sample']} ms/próbkę | {summary['samples_per_sec']:,.1f} próbek/s)")
    if labels is not None:
        print(f"3. Czułość (Recall) : {summary['recall_phishing'] * 100:.2f}% "
              f"(Precision: {summary['precision_phishing'] * 100:.2f}% | F1: {summary['f1_phishing'] * 100:.2f}%)")
        print(f"   Macierz pomyłek [[TN, FP], [FN, TP]]: {summary['confusion_matrix']}")
    print("=" * 64)

    if labels is not None:
        print("\nRaport klasyfikacji:")
        print(classification_report(y_true, preds, target_names=["Benign (0)", "Phishing (1)"], digits=4))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summary,
            "predictions": [
                {
                    "index": i,
                    "prediction": int(p),
                    "label": "PHISHING" if int(p) == 1 else "BENIGN",
                    "phishing_probability": round(float(pr), 6),
                    **({"true_label": int(labels[i])} if labels is not None else {}),
                }
                for i, (p, pr) in enumerate(zip(preds, probas))
            ],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Zapisano szczegółowe predykcje do: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Lokalny skrypt predykcyjny dla phishing_detector.ipynb (Opcja A)")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/phishing_detector_option_a.joblib"),
        help="Ścieżka do pliku modelu .joblib",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Opcjonalny próg decyzyjny (0.0 - 1.0)")
    parser.add_argument("--text", type=str, default=None, help="Pojedynczy URL / treść / HTML do oceny")
    parser.add_argument("--file", type=Path, default=None, help="Ścieżka do pliku HTML/TXT do oceny")
    parser.add_argument("--input", type=Path, default=None, help="Plik JSON [{'text': ..., 'label': ...}]")
    parser.add_argument("--output", type=Path, default=None, help="Plik JSON do zapisu wyników")
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Ewaluacja na 20% zbiorze testowym z combined_reduced.json (random_state=42)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/combined_reduced.json"),
        help="Ścieżka do lokalnego pliku combined_reduced.json",
    )
    args = parser.parse_args()

    detector = LocalPhishingDetector(model_path=args.model, threshold=args.threshold)

    if args.text is not None:
        print(json.dumps(detector.inspect_single(args.text), indent=2, ensure_ascii=False))
        return

    if args.file is not None:
        content = args.file.read_text(encoding="utf-8", errors="ignore")
        print(json.dumps(detector.inspect_single(content), indent=2, ensure_ascii=False))
        return

    if args.input is not None:
        with open(args.input, "r", encoding="utf-8") as f:
            raw = json.load(f)
        texts = [r["text"] if isinstance(r, dict) else str(r) for r in raw]
        labels = [int(r["label"]) for r in raw] if (raw and isinstance(raw[0], dict) and "label" in raw[0]) else None
        evaluate_samples(detector, texts, labels=labels, output_path=args.output)
        return

    if args.eval_test:
        if not args.data_path.exists():
            print(f"[INFO] Pobieranie {DATASET_URL} do {args.data_path}...")
            args.data_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(DATASET_URL, args.data_path)
        with open(args.data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        df = pd.DataFrame({"text": [x["text"] for x in raw], "label": [int(x["label"]) for x in raw]})
        _, test_df = train_test_split(df, test_size=0.2, random_state=42)
        evaluate_samples(
            detector,
            test_df["text"].tolist(),
            labels=test_df["label"].tolist(),
            output_path=args.output,
        )
        return

    # Domyślnie: uruchom test na zapisanej paczce próbek testowych, jeśli istnieje
    for candidate in [Path("data/test_sample_100.json"), Path("data/sample_test.json")]:
        if candidate.exists():
            print(f"[INFO] Uruchamiam ewaluację na paczce próbek: {candidate}")
            with open(candidate, "r", encoding="utf-8") as f:
                raw = json.load(f)
            evaluate_samples(
                detector,
                [r["text"] for r in raw],
                labels=[int(r["label"]) for r in raw],
                output_path=args.output,
            )
            return

    parser.print_help()


if __name__ == "__main__":
    main()
