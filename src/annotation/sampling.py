"""
Sample a 50-turn gold set for hand annotation.

Stratified by dialogue outcome to ensure coverage of all confusion levels.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from src import config

GOLD_SAMPLING_PLAN = {
    "Yes": 20,
    "Yes, but I had to reveal the answer": 20,
    "No": 10,
}
RANDOM_SEED = 42
MIN_DIALOGUE_TURNS = 4
MIN_TEXT_LEN = 10
MAX_TEXT_LEN = 1000


def build_gold_pool() -> pd.DataFrame:
    if not config.MATHDIAL_PROCESSED_TURNS_FILE.exists():
        raise RuntimeError("Run preprocess.py first.")

    turns = pd.read_parquet(config.MATHDIAL_PROCESSED_TURNS_FILE)
    student_turns = turns[turns["speaker"] == "student"].copy()

    text_lens = student_turns["text"].str.len()
    student_turns = student_turns[
        (text_lens >= MIN_TEXT_LEN) & (text_lens <= MAX_TEXT_LEN)
    ]

    dialogue_sizes = turns.groupby("dialogue_id").size()
    big_enough = dialogue_sizes[dialogue_sizes >= MIN_DIALOGUE_TURNS].index
    student_turns = student_turns[student_turns["dialogue_id"].isin(big_enough)]

    parts = []
    for outcome, n in GOLD_SAMPLING_PLAN.items():
        pool = student_turns[student_turns["self_correctness"] == outcome]
        if len(pool) < n:
            print(f"WARNING: '{outcome}' only has {len(pool)} turns, taking all.")
            sampled = pool
        else:
            sampled = pool.sample(n=n, random_state=RANDOM_SEED)
        parts.append(sampled)

    pool_df = pd.concat(parts, ignore_index=True)
    pool_df = pool_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    pool_df["pool_position"] = range(len(pool_df))
    return pool_df


if __name__ == "__main__":
    pool = build_gold_pool()
    print(f"Gold pool size: {len(pool)}")
    print(f"\nOutcome distribution:")
    print(pool["self_correctness"].value_counts())
    out = config.ANNOTATED_DIR / "lsi_gold_pool.parquet"
    pool.to_parquet(out, index=False)
    print(f"\nSaved to {out}")