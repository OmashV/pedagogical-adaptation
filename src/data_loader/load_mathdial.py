"""
Download MathDial from Hugging Face and save locally.

Run once. Subsequent runs will use the cached local file unless
force_download=True.
"""

from datasets import load_dataset
import pandas as pd
from src import config


def download_mathdial(force_download: bool = False) -> pd.DataFrame:
    """
    Download MathDial train split from Hugging Face and save as a
    Parquet file at config.MATHDIAL_LOCAL_RAW_FILE.

    Returns the downloaded dataframe.
    """
    config.ensure_dirs()

    target = config.MATHDIAL_LOCAL_RAW_FILE
    if target.exists() and not force_download:
        print(f"Already downloaded at {target}. Loading from cache.")
        return pd.read_parquet(target)

    if config.HF_TOKEN is None:
        raise RuntimeError(
            "HF_TOKEN not found. Check your .env file at the project root."
        )

    print(f"Downloading {config.MATHDIAL_HF_ID} from Hugging Face...")
    ds = load_dataset(config.MATHDIAL_HF_ID, token=config.HF_TOKEN)

    print(f"Train rows: {len(ds['train'])}, Test rows: {len(ds['test'])}")

    # We use the train split for V1 work. The test split is held back.
    df_train = ds["train"].to_pandas()
    df_train.to_parquet(target, index=False)
    print(f"Saved train split to {target}")

    # Save test split alongside for later use
    test_target = target.parent / "mathdial_raw_test.parquet"
    ds["test"].to_pandas().to_parquet(test_target, index=False)
    print(f"Saved test split to {test_target}")

    return df_train


if __name__ == "__main__":
    df = download_mathdial()
    print(f"\nLoaded {len(df)} dialogues.")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst dialogue qid: {df.iloc[0]['qid']}")
    print(f"Conversation snippet (first 400 chars):")
    print(df.iloc[0]['conversation'][:400])