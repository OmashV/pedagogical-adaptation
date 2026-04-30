"""
Compare distant labels vs gold labels on the 50 gold turns.

This validates how well the distant supervision matches human judgment.
Reports Cohen's kappa and per-class agreement.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from sklearn.metrics import cohen_kappa_score, classification_report
from src import config

GOLD_FILE = config.ANNOTATED_DIR / "lsi_gold_labels.csv"
DISTANT_FILE = config.ANNOTATED_DIR / "lsi_distant_labels.csv"


def main():
    gold = pd.read_csv(GOLD_FILE)
    distant = pd.read_csv(DISTANT_FILE)

    # Match each gold row to its distant counterpart by (dialogue_id, turn_idx)
    merged = gold.merge(
        distant[["dialogue_id", "turn_idx", "confusion_type"]],
        on=["dialogue_id", "turn_idx"],
        suffixes=("_gold", "_distant"),
        how="inner",
    )
    print(f"Matched {len(merged)} gold turns with distant labels.")
    if len(merged) < len(gold):
        print(
            f"  ({len(gold) - len(merged)} gold turns had no distant match — "
            "their dialogues weren't in the distant subsample.)"
        )

    if len(merged) == 0:
        print("\nNo overlap. The 50 gold turns came from dialogues outside the distant subsample.")
        print("This is fine: gold serves as held-out evaluation, but agreement cannot be computed.")
        print("Recommend: re-sample gold from within the distant subsample for V2.")
        return

    print("\n=== Confusion type agreement ===")
    kappa = cohen_kappa_score(merged["confusion_type_gold"], merged["confusion_type_distant"])
    raw_agree = (merged["confusion_type_gold"] == merged["confusion_type_distant"]).mean()
    print(f"Raw agreement: {raw_agree:.3f}")
    print(f"Cohen's kappa: {kappa:.3f}")
    print("\nClassification report (gold = truth, distant = prediction):")
    print(classification_report(
        merged["confusion_type_gold"],
        merged["confusion_type_distant"],
        zero_division=0,
    ))


if __name__ == "__main__":
    main()