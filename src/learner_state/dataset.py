"""
Prepare LSI training datasets from distant labels + gold labels.

Outputs:
- data/processed/lsi_train.parquet  (~5,400 rows from distant labels)
- data/processed/lsi_val.parquet    (~1,300 rows from distant labels)
- data/processed/lsi_test.parquet   (50 rows: gold human labels)

Splits are dialogue-aware — turns from the same dialogue go in the same
split (no leakage).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from src import config

DISTANT_LABELS_FILE = config.ANNOTATED_DIR / "lsi_distant_labels.csv"
GOLD_LABELS_FILE = config.ANNOTATED_DIR / "lsi_gold_labels.csv"

TRAIN_OUT = config.PROCESSED_DIR / "lsi_train.parquet"
VAL_OUT = config.PROCESSED_DIR / "lsi_val.parquet"
TEST_OUT = config.PROCESSED_DIR / "lsi_test.parquet"

VAL_FRACTION = 0.20
RANDOM_SEED = 42


def load_inputs():
    if not DISTANT_LABELS_FILE.exists():
        raise RuntimeError("Run derive_distant_labels.py first.")
    if not GOLD_LABELS_FILE.exists():
        raise RuntimeError("Annotate the 50 gold turns first.")

    distant = pd.read_csv(DISTANT_LABELS_FILE)
    gold = pd.read_csv(GOLD_LABELS_FILE)

    # Standardize columns across the two sources
    distant_cols = [
        "dialogue_id", "turn_idx", "student_text", "context_window",
        "confusion_type", "misconception_flag", "label_source",
    ]
    gold_cols = distant_cols  # gold has these too

    distant = distant[distant_cols]
    gold = gold[gold_cols]

    return distant, gold


def remove_gold_dialogues_from_distant(distant: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Prevent leakage: remove distant rows whose dialogue appears in gold."""
    gold_dialogues = set(gold["dialogue_id"].unique())
    before = len(distant)
    cleaned = distant[~distant["dialogue_id"].isin(gold_dialogues)]
    after = len(cleaned)
    print(
        f"Leakage removal: dropped {before - after} distant rows "
        f"from {len(gold_dialogues)} dialogues that appear in gold."
    )
    return cleaned


def dialogue_aware_train_val_split(distant: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by dialogue_id so all turns from one dialogue go to one split."""
    np.random.seed(RANDOM_SEED)
    unique_dialogues = distant["dialogue_id"].unique()
    np.random.shuffle(unique_dialogues)

    val_count = int(len(unique_dialogues) * VAL_FRACTION)
    val_ids = set(unique_dialogues[:val_count])
    train_ids = set(unique_dialogues[val_count:])

    train = distant[distant["dialogue_id"].isin(train_ids)].copy()
    val = distant[distant["dialogue_id"].isin(val_ids)].copy()

    return train, val


def report_distribution(name: str, df: pd.DataFrame):
    print(f"\n{name}: {len(df)} rows, {df['dialogue_id'].nunique()} dialogues")
    print("  confusion_type:")
    for label, count in df["confusion_type"].value_counts().items():
        print(f"    {label:12s}: {count:5d} ({100*count/len(df):.1f}%)")


def main():
    config.ensure_dirs()

    distant, gold = load_inputs()
    print(f"Loaded {len(distant)} distant labels, {len(gold)} gold labels.")

    distant = remove_gold_dialogues_from_distant(distant, gold)

    train, val = dialogue_aware_train_val_split(distant)
    test = gold

    report_distribution("TRAIN", train)
    report_distribution("VAL", val)
    report_distribution("TEST (gold)", test)

    train.to_parquet(TRAIN_OUT, index=False)
    val.to_parquet(VAL_OUT, index=False)
    test.to_parquet(TEST_OUT, index=False)

    print(f"\nSaved to {config.PROCESSED_DIR}/")


if __name__ == "__main__":
    main()