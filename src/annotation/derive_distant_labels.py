"""
Distant supervision for LSI labels using Groq's free-tier API.

Strategy:
- Subsample 1,000 dialogues stratified by outcome (fits in one day's RPD).
- For each dialogue, classify teacher_described_confusion via Groq Llama.
- Set misconception_flag from student_incorrect_solution presence.
- Expand to all student turns within those dialogues.

Resumable: if interrupted, re-running picks up where it left off using
data/annotated/lsi_distant_dialogue_labels.csv as the checkpoint.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import time
import uuid
from datetime import datetime, timezone

import pandas as pd
from groq import Groq, RateLimitError
from tqdm import tqdm

from src import config

DIALOGUE_LABELS_FILE = config.ANNOTATED_DIR / "lsi_distant_dialogue_labels.csv"
TURN_LABELS_FILE = config.ANNOTATED_DIR / "lsi_distant_labels.csv"

SYSTEM_PROMPT = """You are an expert annotator classifying student confusion in math tutoring dialogues.

Read a teacher's free-text description of why a student got a math problem wrong, and classify it into exactly one of four categories.

CRITICAL DISTINCTION: "procedural" means the student set up the problem CORRECTLY and only made an arithmetic slip during calculation. If the student set up the problem WRONG (used the wrong fraction, double-counted something, applied the wrong operation, used the wrong quantity), that is "conceptual", not procedural. Most errors involving "wrong total", "wrong fraction", "double counting", "missed a step", or "treated X as Y" are conceptual setup errors, not procedural calculation errors.

Categories:

- "none": The description does not indicate a real confusion. Either the student was on track, OR the description is too vague/short to identify a specific error type (e.g., "made a mistake", "got it wrong", "went too far" with no further detail). When in doubt and the description is uninformative, choose "none".

- "lexical": The student's confusion is specifically about the meaning of a word, term, label, or notation in the problem. Examples: misreading "tripled" as "added three", confusing "before midterms" with "during midterms", misinterpreting what a unit name refers to. The student would understand the math if the word were redefined.

- "conceptual": The student has a wrong mental model or wrong setup of the problem. The error is structural and would not be fixed by re-checking arithmetic. Examples: applying the wrong operation, treating part as whole (e.g., "took 4/4 instead of 4/5"), double counting, using the wrong total quantity, missing what information is relevant, treating area as volume, treating rate as total. If the description mentions the student "took X as Y", "used the wrong [quantity/fraction/operation]", or "didn't account for [something]" — that is conceptual.

- "procedural": The student understood the concept and set up the problem correctly, but made a local arithmetic slip during execution. Examples: dropped a digit, mis-multiplied, forgot to carry, miscounted by one. The student would get it right with a calculator. Use this ONLY when the description clearly points to a calculation slip, not a setup error.

Decision steps:
1. Is the description too vague or generic to pin down an error type? → "none"
2. Is the confusion about what a word/term/unit means? → "lexical"
3. Is the error in setup, model, or framing of the problem? → "conceptual"
4. Is the error purely a calculation slip with correct setup? → "procedural"

Rationale must be specific to THIS description. Do not just restate the category definition. Quote or paraphrase the specific error from the input.

Output ONLY a JSON object, no other text:
{"confusion_type": "none" | "lexical" | "conceptual" | "procedural", "rationale": "<one short specific sentence>"}"""

USER_PROMPT_TEMPLATE = """Teacher's description of student confusion:

\"\"\"
{description}
\"\"\"

Respond with the JSON object only."""


# ----- Sampling -----

def stratified_subsample_dialogues(turns: pd.DataFrame) -> pd.DataFrame:
    """Pick ~1,000 dialogues stratified by outcome."""
    dialogues = turns.drop_duplicates("dialogue_id")[
        ["dialogue_id", "self_correctness",
         "teacher_described_confusion", "student_incorrect_solution"]
    ].reset_index(drop=True)

    parts = []
    for outcome, n in config.DISTANT_SAMPLE_PLAN.items():
        pool = dialogues[dialogues["self_correctness"] == outcome]
        if len(pool) < n:
            print(f"  '{outcome}': only {len(pool)} available, taking all.")
            sampled = pool
        else:
            sampled = pool.sample(n=n, random_state=config.DISTANT_RANDOM_SEED)
        parts.append(sampled)
    out = pd.concat(parts, ignore_index=True)
    print(f"Subsampled {len(out)} dialogues.")
    return out


# ----- Classification -----

def classify_one(client: Groq, description: str) -> dict:
    """Call Groq once. Retry once on rate limit."""
    if not description or not isinstance(description, str) or not description.strip():
        return {"confusion_type": "none", "rationale": "empty description"}

    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": USER_PROMPT_TEMPLATE.format(description=description.strip())},
                ],
                max_tokens=200,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            return parse_response(raw)
        except RateLimitError:
            if attempt == 0:
                print("\n  Rate limited — sleeping 60s and retrying.")
                time.sleep(60)
            else:
                return {"confusion_type": "none", "rationale": "RATE LIMIT FAIL"}
        except Exception as e:
            return {"confusion_type": "none", "rationale": f"API ERROR: {str(e)[:80]}"}


def parse_response(raw: str) -> dict:
    """Parse the LLM's JSON output, with fallbacks for malformed responses."""
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                result = json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                return {"confusion_type": "none", "rationale": f"PARSE FAIL: {raw[:80]}"}
        else:
            return {"confusion_type": "none", "rationale": f"PARSE FAIL: {raw[:80]}"}

    valid = {"none", "lexical", "conceptual", "procedural"}
    if result.get("confusion_type") not in valid:
        return {"confusion_type": "none", "rationale": f"INVALID: {str(result)[:80]}"}
    return result


# ----- Main loop with checkpoint/resume -----

def run_classification(dialogues_to_label: pd.DataFrame, client: Groq) -> pd.DataFrame:
    """Classify each dialogue, saving progress every 50 calls."""
    # Resume: load any existing checkpoint
    if DIALOGUE_LABELS_FILE.exists():
        existing = pd.read_csv(DIALOGUE_LABELS_FILE)
        done_ids = set(existing["dialogue_id"].tolist())
        print(f"Resume: {len(done_ids)} dialogues already classified.")
    else:
        existing = pd.DataFrame()
        done_ids = set()

    todo = dialogues_to_label[~dialogues_to_label["dialogue_id"].isin(done_ids)]
    print(f"To classify: {len(todo)} dialogues.")
    if len(todo) == 0:
        return existing

    new_results = []
    pbar = tqdm(todo.iterrows(), total=len(todo), desc="Groq calls")
    for i, (_, row) in enumerate(pbar):
        result = classify_one(client, row["teacher_described_confusion"])
        misc_flag = 1 if (
            isinstance(row["student_incorrect_solution"], str)
            and row["student_incorrect_solution"].strip()
        ) else 0

        new_results.append({
            "dialogue_id": row["dialogue_id"],
            "confusion_type": result["confusion_type"],
            "llm_rationale": result["rationale"],
            "misconception_flag": misc_flag,
        })

        # Rate limit pacing
        time.sleep(config.GROQ_REQUEST_DELAY_SEC)

        # Checkpoint every 50
        if (i + 1) % 50 == 0:
            checkpoint = pd.concat(
                [existing, pd.DataFrame(new_results)], ignore_index=True
            )
            checkpoint.to_csv(DIALOGUE_LABELS_FILE, index=False)

    final = pd.concat([existing, pd.DataFrame(new_results)], ignore_index=True)
    final.to_csv(DIALOGUE_LABELS_FILE, index=False)
    return final


def expand_to_turn_level(turns: pd.DataFrame, dlg_labels: pd.DataFrame) -> pd.DataFrame:
    student_turns = turns[turns["speaker"] == "student"].copy()
    student_turns = student_turns[
        student_turns["dialogue_id"].isin(dlg_labels["dialogue_id"])
    ]
    merged = student_turns.merge(dlg_labels, on="dialogue_id", how="left")

    now = datetime.now(timezone.utc).isoformat()
    merged["annotation_id"] = [str(uuid.uuid4()) for _ in range(len(merged))]
    merged["label_source"] = "distant_supervision"
    merged["annotated_at"] = now
    merged["annotation_round"] = 1
    merged["annotator_notes"] = ""
    merged["skip_reason"] = ""

    keep = [
        "annotation_id", "dialogue_id", "turn_idx",
        "text", "context_window",
        "confusion_type", "misconception_flag",
        "llm_rationale", "label_source",
        "annotator_notes", "skip_reason",
        "annotated_at", "annotation_round",
    ]
    return merged[keep].rename(columns={"text": "student_text"})


def main(limit: int | None = None):
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not in .env")
    if not config.MATHDIAL_PROCESSED_TURNS_FILE.exists():
        raise RuntimeError("Run preprocess.py first.")

    config.ensure_dirs()
    turns = pd.read_parquet(config.MATHDIAL_PROCESSED_TURNS_FILE)

    dialogues_to_label = stratified_subsample_dialogues(turns)
    if limit:
        dialogues_to_label = dialogues_to_label.head(limit)
        print(f"DEV MODE: limit={limit}")

    client = Groq(api_key=config.GROQ_API_KEY)

    print(f"\nUsing model: {config.GROQ_MODEL}")
    print(f"Estimated wall time at 30 RPM: ~{len(dialogues_to_label) / 30:.1f} min\n")

    dlg_labels = run_classification(dialogues_to_label, client)

    print(f"\n=== Dialogue-level distribution ===")
    print(dlg_labels["confusion_type"].value_counts())
    print(f"\n=== Misconception flag distribution ===")
    print(dlg_labels["misconception_flag"].value_counts())

    turn_labels = expand_to_turn_level(turns, dlg_labels)
    turn_labels.to_csv(TURN_LABELS_FILE, index=False)
    print(f"\nSaved {len(turn_labels)} turn-level labels to {TURN_LABELS_FILE}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="Limit dialogues for testing.")
    args = p.parse_args()
    main(limit=args.limit)