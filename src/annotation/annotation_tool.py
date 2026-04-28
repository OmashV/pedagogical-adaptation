"""
Streamlit annotation tool for the LSI gold set (50 turns).

Run: streamlit run src/annotation/annotation_tool.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src import config

GOLD_POOL_FILE = config.ANNOTATED_DIR / "lsi_gold_pool.parquet"
GOLD_LABELS_FILE = config.ANNOTATED_DIR / "lsi_gold_labels.csv"

CONFUSION_OPTIONS = ["none", "lexical", "conceptual", "procedural"]
MISCONCEPTION_OPTIONS = [0, 1]
SKIP_OPTIONS = ["", "broken_dialogue", "non_pedagogical", "ambiguous", "other"]


@st.cache_data
def load_pool() -> pd.DataFrame:
    if not GOLD_POOL_FILE.exists():
        st.error("Run `python -m src.annotation.sampling` first.")
        st.stop()
    return pd.read_parquet(GOLD_POOL_FILE)


def load_labels() -> pd.DataFrame:
    if GOLD_LABELS_FILE.exists():
        return pd.read_csv(GOLD_LABELS_FILE)
    cols = [
        "annotation_id", "dialogue_id", "turn_idx", "pool_position",
        "student_text", "context_window",
        "confusion_type", "misconception_flag",
        "label_source", "annotator_notes", "skip_reason",
        "annotated_at", "annotation_round",
    ]
    return pd.DataFrame(columns=cols)


def save_label(record: dict):
    df = load_labels()
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(GOLD_LABELS_FILE, index=False)


def already_done(round_id: int = 1) -> set:
    df = load_labels()
    if df.empty:
        return set()
    return set(df[df["annotation_round"] == round_id]["pool_position"].tolist())


def main():
    st.set_page_config(page_title="LSI Gold Annotation", layout="wide")
    st.title("LSI Gold Annotation Tool")
    st.caption("50 hand-labeled turns. Refer to docs/05_annotation_guide.md.")

    pool = load_pool()
    done = already_done(round_id=1)
    remaining = [p for p in pool["pool_position"].tolist() if p not in done]

    with st.sidebar:
        st.metric("Annotated", f"{len(done)} / {len(pool)}")
        st.progress(len(done) / max(len(pool), 1))

        if not remaining:
            st.success("Round 1 complete!")
            st.stop()

        target_position = st.number_input(
            "Jump to pool position:",
            min_value=0, max_value=len(pool) - 1,
            value=remaining[0], step=1,
        )

        st.divider()
        st.subheader("Quick legend")
        st.markdown(
            "- **none** — no confusion\n"
            "- **lexical** — word/term confusion\n"
            "- **conceptual** — wrong mental model\n"
            "- **procedural** — execution error\n\n"
            "**misconception_flag**:\n"
            "- 0 — no stable wrong model\n"
            "- 1 — stable wrong model present"
        )

    row = pool[pool["pool_position"] == target_position].iloc[0]

    st.subheader(f"Turn {target_position + 1} of {len(pool)}")
    st.caption(
        f"dialogue_id: `{row['dialogue_id']}` | turn_idx: {row['turn_idx']} | "
        f"outcome: `{row['self_correctness']}`"
    )

    with st.expander("📘 Math problem", expanded=False):
        st.write(row["question"])
        st.markdown("**Ground truth:**")
        st.write(row.get("ground_truth", "—"))
        st.markdown("**Seeded incorrect solution:**")
        st.write(row.get("student_incorrect_solution", "—"))
        st.markdown("**Teacher's description of confusion:**")
        st.write(row.get("teacher_described_confusion", "—"))

    with st.expander("🧵 Context (last up to 4 turns)", expanded=True):
        ctx = row.get("context_window", "")
        if ctx:
            for line in ctx.split(" || "):
                st.write(line)
        else:
            st.write("_(first student turn)_")

    st.divider()
    st.subheader("👇 Student turn to label")
    st.info(row["text"])

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        confusion_type = st.radio("confusion_type", CONFUSION_OPTIONS, index=0)
    with col2:
        misconception_flag = st.radio(
            "misconception_flag", MISCONCEPTION_OPTIONS, index=0,
            format_func=lambda x: f"{x} — {'misconception present' if x == 1 else 'no stable misconception'}"
        )

    annotator_notes = st.text_area("annotator_notes (optional)")
    skip_reason = st.selectbox("skip_reason (only if skipping)", SKIP_OPTIONS)

    if st.button("✅ Save and next", type="primary", use_container_width=True):
        record = {
            "annotation_id": str(uuid.uuid4()),
            "dialogue_id": row["dialogue_id"],
            "turn_idx": int(row["turn_idx"]),
            "pool_position": int(target_position),
            "student_text": row["text"],
            "context_window": row.get("context_window", ""),
            "confusion_type": confusion_type,
            "misconception_flag": int(misconception_flag),
            "label_source": "gold_human",
            "annotator_notes": annotator_notes,
            "skip_reason": skip_reason or "",
            "annotated_at": datetime.now(timezone.utc).isoformat(),
            "annotation_round": 1,
        }
        save_label(record)
        st.success(f"Saved position {target_position}.")
        st.rerun()


if __name__ == "__main__":
    main()