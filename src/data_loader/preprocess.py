"""
Parse MathDial dialogues into per-turn rows.

Each row in the output represents a single turn in a dialogue with:
- dialogue-level metadata (qid, question, ground_truth, ...)
- turn-level fields (turn_idx, speaker, move, text)
- context (preceding turns within the same dialogue)

The output is saved to config.MATHDIAL_PROCESSED_TURNS_FILE.
"""

import re
import pandas as pd
from src import config

# Format observed in MathDial conversations:
#   Speaker: (move) text |EOM| Speaker: (move) text |EOM| ...
# Speaker is "Teacher" or "Student". Move is a short tag like "focus",
# "probing", "telling", "generic".
TURN_PATTERN = re.compile(
    r"^\s*(Teacher|Student)\s*:\s*(?:\(([^)]+)\))?\s*(.*)$",
    re.DOTALL,
)


def parse_conversation(conversation: str) -> list[dict]:
    """Split the |EOM|-delimited conversation string into turn dicts."""
    if not isinstance(conversation, str) or not conversation.strip():
        return []

    raw_turns = [t.strip() for t in conversation.split("|EOM|") if t.strip()]
    parsed = []
    for idx, raw in enumerate(raw_turns):
        m = TURN_PATTERN.match(raw)
        if not m:
            # Skip malformed turns rather than crashing
            continue
        speaker, move, text = m.groups()
        parsed.append({
            "turn_idx": idx,
            "speaker": speaker.lower(),
            "move": (move or "").strip().lower() or None,
            "text": text.strip(),
        })
    return parsed


def explode_to_turns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert dialogue-level dataframe to turn-level dataframe.

    Each row in the input dataframe represents ONE dialogue (one student's
    session on one problem). Note that qid is NOT unique — the same problem
    can be given to multiple students. We assign a unique dialogue_id by
    combining qid with the row index in the original dataframe.

    Each turn row carries:
    - dialogue_id (unique per dialogue), qid (problem id), question, ...
    - turn_idx, speaker, move, text
    - context_window: text of up to 4 preceding turns
    """
    rows = []
    for row_idx, dlg in df.iterrows():
        # Build a unique dialogue id: combine qid with the row index
        dialogue_id = f"{dlg['qid']}_{row_idx}"

        turns = parse_conversation(dlg["conversation"])
        for t in turns:
            ctx = [
                f"{tt['speaker'].capitalize()}: {tt['text']}"
                for tt in turns[max(0, t["turn_idx"] - 4):t["turn_idx"]]
            ]
            rows.append({
                "dialogue_id": dialogue_id,
                "qid": dlg["qid"],
                "question": dlg["question"],
                "ground_truth": dlg.get("ground_truth"),
                "student_incorrect_solution": dlg.get(
                    "student_incorrect_solution"
                ),
                "student_profile": dlg.get("student_profile"),
                "teacher_described_confusion": dlg.get(
                    "teacher_described_confusion"
                ),
                "self_correctness": dlg.get("self-correctness"),
                "turn_idx": t["turn_idx"],
                "speaker": t["speaker"],
                "move": t["move"],
                "text": t["text"],
                "context_window": " || ".join(ctx) if ctx else "",
            })
    return pd.DataFrame(rows)


def preprocess_and_save(force: bool = False) -> pd.DataFrame:
    """Run the full preprocess pipeline. Cached unless force=True."""
    config.ensure_dirs()
    target = config.MATHDIAL_PROCESSED_TURNS_FILE
    if target.exists() and not force:
        print(f"Already processed at {target}. Loading from cache.")
        return pd.read_parquet(target)

    if not config.MATHDIAL_LOCAL_RAW_FILE.exists():
        raise RuntimeError(
            "Raw MathDial not found. Run "
            "`python -m src.data_loader.load_mathdial` first."
        )

    df = pd.read_parquet(config.MATHDIAL_LOCAL_RAW_FILE)
    print(f"Loaded {len(df)} dialogues.")
    turns = explode_to_turns(df)
    print(f"Exploded to {len(turns)} turns.")
    turns.to_parquet(target, index=False)
    print(f"Saved to {target}")
    return turns


if __name__ == "__main__":
    turns = preprocess_and_save(force=True)
    print(f"\nTotal turns: {len(turns)}")
    print(f"Total dialogues (unique dialogue_id): {turns['dialogue_id'].nunique()}")
    print(f"Unique qids (problems): {turns['qid'].nunique()}")
    print(f"\nSpeaker distribution:\n{turns['speaker'].value_counts()}")
    print(f"\nMove distribution (teacher only):")
    print(turns[turns['speaker'] == 'teacher']['move'].value_counts())