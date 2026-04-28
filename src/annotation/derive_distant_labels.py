"""
Distant supervision for LSI labels on MathDial.

For each dialogue:
  - Read the teacher's description of student confusion
  - Classify it into {none, lexical, conceptual, procedural} via Claude
  - Set misconception_flag = 1 if student_incorrect_solution is present
  - Propagate the label to every student turn within that dialogue

Output: data/annotated/lsi_distant_labels.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import time
import uuid
from datetime import datetime, timezone

import pandas as pd
from anthropic import Anthropic
from tqdm import tqdm

from src import config

# ----- Configuration -----

CLAUDE_MODEL = "claude-haiku-4-5-20251001"  # cheap classification model
SYSTEM_PROMPT = """You are an expert annotator classifying student confusion in math tutoring dialogues.

Your task: read a teacher's free-text description of why a student got a math problem wrong, and classify it into exactly one of four categories.

Categories:

- "none": The description does not indicate confusion (e.g., the student was on track or made no error worth noting).
- "lexical": The student's confusion is about the meaning of a specific word, term, or notation in the problem (e.g., misreading "tripled" as "added three", misinterpreting "the rest", confusing "before midterms").
- "conceptual": The student has a wrong or incomplete mental model of the underlying mathematical concept. This is structural confusion that persists — wrong operation, wrong relationship between quantities, applying the wrong concept entirely.
- "procedural": The student understands the concept but stumbles in execution — arithmetic slip, miscount, dropped digit, calculation error. The mistake is local and would be fixed by a careful re-check.

Decision rule:
1. If no real confusion is described → "none"
2. If confusion is about a specific word or term meaning → "lexical"
3. If confusion is about a wrong mental model or structural misunderstanding → "conceptual"
4. If confusion is about execution mistakes (arithmetic, miscounting) → "procedural"

You must output ONLY a JSON object with this exact format and nothing else:
{"confusion_type": "none" | "lexical" | "conceptual" | "procedural", "rationale": "<one short sentence>"}
"""

USER_PROMPT_TEMPLATE = """Teacher's description of student confusion:

\"\"\"
{description}
\"\"\"

Classify into one of: none, lexical, conceptual, procedural.

Respond with the JSON object only."""


# ----- Core functions -----

def classify_confusion_description(client: Anthropic, description: str) -> dict:
    """Call Claude to classify a single teacher confusion description."""
    if not description or not isinstance(description, str) or not description.strip():
        return {"confusion_type": "none", "rationale": "empty description"}

    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(description=description.strip())}
        ],
    )
    raw = msg.content[0].text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback parsing — find the first { ... } block
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                result = json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                return {"confusion_type": "none", "rationale": f"PARSE FAIL: {raw[:80]}"}
        else:
            return {"confusion_type": "none", "rationale": f"PARSE FAIL: {raw[:80]}"}

    valid = {"none", "lexical", "conceptual", "procedural"}
    if result.get("confusion_type") not in valid:
        result = {"confusion_type": "none", "rationale": f"INVALID LABEL: {result}"}
    return result


def derive_dialogue_level_labels(turns: pd.DataFrame, client: Anthropic) -> pd.DataFrame:
    """
    For each unique dialogue, classify the teacher's confusion description.
    Return a dataframe with columns: dialogue_id, confusion_type,
    misconception_flag, llm_rationale.
    """
    dialogues = (
        turns[["dialogue_id", "teacher_described_confusion",
               "student_incorrect_solution"]]
        .drop_duplicates("dialogue_id")
        .reset_index(drop=True)
    )
    print(f"Classifying {len(dialogues)} dialogue confusion descriptions...")

    results = []
    for _, row in tqdm(dialogues.iterrows(), total=len(dialogues)):
        desc = row["teacher_described_confusion"]
        classification = classify_confusion_description(client, desc)
        misc_flag = 1 if (
            isinstance(row["student_incorrect_solution"], str)
            and row["student_incorrect_solution"].strip()
        ) else 0
        results.append({
            "dialogue_id": row["dialogue_id"],
            "confusion_type": classification["confusion_type"],
            "llm_rationale": classification["rationale"],
            "misconception_flag": misc_flag,
        })
        # Light rate limit to be polite
        time.sleep(0.05)

    return pd.DataFrame(results)


def expand_to_turn_level(turns: pd.DataFrame, dialogue_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Join dialogue-level labels to every STUDENT turn in that dialogue.
    """
    student_turns = turns[turns["speaker"] == "student"].copy()
    merged = student_turns.merge(dialogue_labels, on="dialogue_id", how="left")

    now_iso = datetime.now(timezone.utc).isoformat()
    merged["annotation_id"] = [str(uuid.uuid4()) for _ in range(len(merged))]
    merged["label_source"] = "distant_supervision"
    merged["annotated_at"] = now_iso
    merged["annotation_round"] = 1
    merged["annotator_notes"] = ""
    merged["skip_reason"] = ""

    keep_cols = [
        "annotation_id", "dialogue_id", "turn_idx",
        "text", "context_window",
        "confusion_type", "misconception_flag",
        "llm_rationale", "label_source",
        "annotator_notes", "skip_reason",
        "annotated_at", "annotation_round",
    ]
    return merged[keep_cols].rename(columns={"text": "student_text"})


# ----- Entry point -----

def main(limit: int | None = None):
    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not in .env — get one from console.anthropic.com"
        )
    if not config.MATHDIAL_PROCESSED_TURNS_FILE.exists():
        raise RuntimeError("Run preprocess.py first.")

    turns = pd.read_parquet(config.MATHDIAL_PROCESSED_TURNS_FILE)

    if limit:
        keep_dialogues = turns["dialogue_id"].unique()[:limit]
        turns = turns[turns["dialogue_id"].isin(keep_dialogues)]
        print(f"DEV MODE: limiting to {limit} dialogues.")

    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

    dlg_labels = derive_dialogue_level_labels(turns, client)
    print(f"\nConfusion-type distribution at dialogue level:")
    print(dlg_labels["confusion_type"].value_counts())
    print(f"\nMisconception flag distribution:")
    print(dlg_labels["misconception_flag"].value_counts())

    turn_labels = expand_to_turn_level(turns, dlg_labels)

    out_path = config.ANNOTATED_DIR / "lsi_distant_labels.csv"
    turn_labels.to_csv(out_path, index=False)
    print(f"\nSaved {len(turn_labels)} turn-level labels to {out_path}")

    dlg_out = config.ANNOTATED_DIR / "lsi_distant_dialogue_labels.csv"
    dlg_labels.to_csv(dlg_out, index=False)
    print(f"Saved {len(dlg_labels)} dialogue-level labels to {dlg_out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit to first N dialogues (for testing)."
    )
    args = parser.parse_args()
    main(limit=args.limit)