"""CLI: train the classical baseline classifier on extracted features.

Two models are fit (both cheap on CPU):
  - logistic regression (calibrated, linear in features)
  - gradient boosting (non-linear, captures feature interactions)

We report the standard anti-spoofing metrics on a held-out test split:
  - ROC-AUC
  - APCER  = FN rate on attacks (fakes misclassified as real)  — lower is better
  - BPCER  = FP rate on bona-fide (reals misclassified as fakes) — lower is better
  - ACER   = (APCER + BPCER) / 2
  - TPR@FPR=1% (operating point useful for production gates)

Usage:
    python -m scripts.train_classical
    python -m scripts.train_classical --features data/features/baseline.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def antispoof_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Standard anti-spoofing metrics.

    y_true : 1 = fake (spoof / positive attack), 0 = real (bona-fide)
    scores : probability of class 1 (fake)
    """
    out: dict = {}
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        # TPR at FPR <= 1%
        idx = np.where(fpr <= 0.01)[0]
        out["tpr_at_fpr1"] = float(tpr[idx[-1]]) if len(idx) else 0.0
        # Threshold that minimises ACER = (FN_fake + FP_real) / 2
        best_acer = 1.0
        best_thr = 0.5
        for thr in thresholds:
            pred = (scores >= thr).astype(np.int64)
            apcer = float(((pred == 0) & (y_true == 1)).sum()) / max(
                1, int((y_true == 1).sum())
            )
            bpcer = float(((pred == 1) & (y_true == 0)).sum()) / max(
                1, int((y_true == 0).sum())
            )
            acer = 0.5 * (apcer + bpcer)
            if acer < best_acer:
                best_acer = acer
                best_thr = float(thr)
                out["apcer"] = apcer
                out["bpcer"] = bpcer
        out["acer"] = best_acer
        out["best_threshold"] = best_thr
    # Point accuracy at 0.5
    pred = (scores >= 0.5).astype(np.int64)
    out["accuracy_at_0.5"] = float((pred == y_true).mean())
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    out["confusion_matrix"] = cm.tolist()  # [[TN,FP],[FN,TP]]
    return out


def make_models() -> dict:
    return {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, C=1.0)),
            ]
        ),
        "gbm": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05
        ),
    }


def report_feature_importance(
    model, feature_names: list[str], top: int = 10
) -> None:
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "named_steps") and hasattr(
        model.named_steps.get("clf"), "coef_"
    ):
        imp = np.abs(model.named_steps["clf"].coef_[0])
    else:
        return
    order = np.argsort(imp)[::-1][:top]
    print("  top features:")
    for i in order:
        print(f"    {feature_names[i]:<22s}  {imp[i]: .4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--features", type=Path, default=Path("data/features/baseline.npz")
    )
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-dir", type=Path, default=Path("data/features")
    )
    args = p.parse_args()

    data = np.load(args.features, allow_pickle=True)
    X, y = data["X"], data["y"]
    feature_names = list(data["feature_names"])
    print(f"loaded features: X={X.shape}, y={y.shape}")
    print(f"class balance: real={int((y==0).sum())}, fake={int((y==1).sum())}")

    if len(np.unique(y)) < 2:
        print(
            "\n[!] Only one class present in the dataset. You need BOTH real "
            "and fake clips to train.\n"
            "    - Add real videos to data/real/raw/ and rerun extract_features.\n"
            "    - Or add source images to data/source_images/ and generate_fakes.\n"
        )
        return

    stratify = y if len(np.unique(y)) > 1 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=stratify
    )

    models = make_models()
    results: dict[str, dict] = {}
    for name, model in models.items():
        print(f"\n=== {name} ===")
        model.fit(X_tr, y_tr)
        scores_te = model.predict_proba(X_te)[:, 1]
        metrics = antispoof_metrics(y_te, scores_te)
        results[name] = metrics
        for k, v in metrics.items():
            if k == "confusion_matrix":
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v:.4f}")
        report_feature_importance(model, feature_names)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.out_dir / f"{name}.joblib")

    # Save a simple report
    (args.out_dir / "classical_report.txt").write_text(
        "\n\n".join(
            f"=== {n} ===\n"
            + "\n".join(
                f"{k}: {v}" for k, v in m.items()
            )
            for n, m in results.items()
        )
    )
    print(f"\nreport -> {args.out_dir / 'classical_report.txt'}")


if __name__ == "__main__":
    main()
