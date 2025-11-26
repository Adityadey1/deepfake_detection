#!/usr/bin/env python3
"""
scripts/eval_table.py

Read metrics for each trained model (efficientnet, mobilenet, resnet)
from results/<model>/test_metrics.json and unseen_metrics.json and
print a comparison table. Also save it as CSV and Markdown.
"""

import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"E:\code\deepfake_project")
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS = ["efficientnet", "mobilenet", "resnet"]  # change if you add more

def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    rows = []

    for name in MODELS:
        model_dir = RESULTS_DIR / name
        test_path = model_dir / "test_metrics.json"
        unseen_path = model_dir / "unseen_metrics.json"

        test = load_json(test_path)
        unseen = load_json(unseen_path)

        if test is None and unseen is None:
            print(f"[WARN] No metrics found for model '{name}' in {model_dir}")
            continue

        row = {"model": name}

        if test is not None:
            # expected keys: loss, accuracy, auc
            row["test_loss"] = test.get("loss")
            row["test_acc"] = test.get("accuracy")
            row["test_auc"] = test.get("auc")
        else:
            row["test_loss"] = None
            row["test_acc"] = None
            row["test_auc"] = None

        if unseen is not None:
            row["unseen_loss"] = unseen.get("loss")
            row["unseen_acc"] = unseen.get("accuracy")
            row["unseen_auc"] = unseen.get("auc")
        else:
            row["unseen_loss"] = None
            row["unseen_acc"] = None
            row["unseen_auc"] = None

        rows.append(row)

    if not rows:
        print("No metrics found for any model. Did you run training first?")
        return

    df = pd.DataFrame(rows)
    # order columns nicely
    df = df[
        [
            "model",
            "test_loss", "test_acc", "test_auc",
            "unseen_loss", "unseen_acc", "unseen_auc",
        ]
    ]

    # sort by test_acc descending (best model first)
    df = df.sort_values(by="test_acc", ascending=False)

    print("\n=== MODEL COMPARISON TABLE ===\n")
    # pretty print with rounding
    display_df = df.copy()
    for col in ["test_loss", "test_acc", "test_auc", "unseen_loss", "unseen_acc", "unseen_auc"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].astype("float64").round(4)

    print(display_df.to_string(index=False))

    # save outputs
    out_dir = RESULTS_DIR / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "model_comparison.csv"
    md_path = out_dir / "model_comparison.md"

    display_df.to_csv(csv_path, index=False)

    # also save markdown table (for report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(display_df.to_markdown(index=False))

    print("\nSaved comparison to:")
    print("  CSV:", csv_path)
    print("  MD :", md_path)

if __name__ == "__main__":
    main()

