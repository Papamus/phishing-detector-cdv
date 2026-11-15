"""Train the compact local phishing filter and evaluate a held-out split."""
import json
import time
import hashlib
import urllib.request
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split

from phishing_detector_phishing_features import build_option_a_pipeline

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "combined_reduced.json"
MODEL = ROOT / "models" / "phishing_detector_option_a.joblib"
OUT = ROOT / "results"
DATASET_URL = "https://huggingface.co/datasets/ealvaradob/phishing-dataset/resolve/main/combined_reduced.json"
ARTIFACT_VERSION = 2

def main():
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        print(f"Pobieranie niezmienionego źródła danych: {DATASET_URL}")
        urllib.request.urlretrieve(DATASET_URL, DATA)
    dataset_sha256 = hashlib.sha256(DATA.read_bytes()).hexdigest()
    rows = json.loads(DATA.read_text(encoding="utf-8"))
    x = np.asarray([r["text"] for r in rows], dtype=object)
    y = np.asarray([int(r["label"]) for r in rows], dtype=np.int8)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42, stratify=y)
    x_fit, x_val, y_fit, y_val = train_test_split(x_train, y_train, test_size=.1, random_state=42, stratify=y_train)
    calibrator = build_option_a_pipeline()
    calibrator.fit(x_fit, y_fit)
    val_prob = calibrator.predict_proba(x_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, val_prob)
    valid = np.flatnonzero(recall[:-1] >= .94)
    threshold = float(thresholds[valid[-1]]) if len(valid) else .5
    model = build_option_a_pipeline()
    model.fit(x_train, y_train)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"artifact_version": ARTIFACT_VERSION, "feature_version": 2, "pipeline": model, "threshold": threshold, "option": "A_ultra_fast_host_features", "dataset_source": "ealvaradob/phishing-dataset:combined_reduced", "dataset_sha256": dataset_sha256, "split_seed": 42, "test_size": .2, "target_validation_recall": .94}, MODEL, compress=3)
    _ = model.predict_proba(x_test[:min(10, len(x_test))])
    start = time.perf_counter()
    prob = model.predict_proba(x_test)[:, 1]
    seconds = time.perf_counter() - start
    pred = (prob >= threshold).astype(int)
    result = {"samples": len(y_test), "threshold": threshold, "recall_phishing": float(recall_score(y_test,pred)), "precision_phishing": float(precision_score(y_test,pred)), "f1_phishing": float(f1_score(y_test,pred)), "inference_seconds": seconds, "ms_per_sample": seconds/len(y_test)*1000, "samples_per_second": len(y_test)/seconds, "model_bytes": MODEL.stat().st_size}
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_path = OUT / "metrics_baseline.json"
    if not baseline_path.exists():
        baseline_path.write_text(json.dumps({"threshold": .5, "recall_phishing": .912245833970341, "precision_phishing": .9512195121951219, "f1_phishing": .931325113157484, "inference_seconds": 8.24837899999693, "ms_per_sample": .5309203784756005, "samples_per_second": 1883.5215986081364, "model_bytes": 247170}, indent=2), encoding="utf-8")
    result["baseline"] = json.loads(baseline_path.read_text(encoding="utf-8"))
    (OUT/"metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    ConfusionMatrixDisplay.from_predictions(y_test,pred,display_labels=["Benign","Phishing"],ax=axes[0],colorbar=False)
    names=["Recall", "Precision", "F1"]
    axes[1].bar(np.arange(3)-.18,[result["baseline"]["recall_phishing"],result["baseline"]["precision_phishing"],result["baseline"]["f1_phishing"]],width=.36,label="Bazowy")
    axes[1].bar(np.arange(3)+.18,[result["recall_phishing"],result["precision_phishing"],result["f1_phishing"]],width=.36,label="Nowy")
    axes[1].set_xticks(np.arange(3),names); axes[1].set_ylim(0,1); axes[1].set_ylabel("Wynik"); axes[1].legend()
    fig.suptitle(f"Test {len(y_test):,} próbek | model {MODEL.stat().st_size/1024:.1f} KB | {result['ms_per_sample']:.3f} ms/próbkę")
    fig.tight_layout(); fig.savefig(OUT/"test_results.png",dpi=160)
    print(json.dumps(result,indent=2))

if __name__ == "__main__": main()
