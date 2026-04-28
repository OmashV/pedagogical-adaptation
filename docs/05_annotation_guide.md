---

## 10. Labeling strategy (Path B + mini-A)

LSI labels are produced via two complementary methods.

### 10.1 Distant supervision (~10,000 turns)

For each MathDial dialogue, the original real teacher provided a free-text
description of the student's confusion in the field
`teacher_described_confusion`. We use Claude as an automated classifier to
map that description onto our four-class confusion_type taxonomy
(none / lexical / conceptual / procedural).

Every student turn within a dialogue inherits its dialogue's classified
confusion_type. The misconception_flag is set to 1 when the dialogue's
`student_incorrect_solution` field is non-empty (i.e., the data collection
process seeded a misconception for this dialogue), and 0 otherwise.

This is **distant supervision** — labels are derived programmatically from
existing expert metadata rather than annotated turn-by-turn. The trade-off
is that labels are dialogue-level (turn variation within a dialogue is
not captured) but the dataset is two orders of magnitude larger than what
hand annotation would produce for one student in two days.

### 10.2 Gold human labels (50 turns)

We hand-annotate a stratified 50-turn gold set under this guide. The
gold set is held out from training and is used for:

- Evaluating the distantly-supervised labels for agreement.
- Evaluating the trained LSI classifier (held-out test).
- Detecting systematic disagreement between human, LLM, and distant
  labels (which would indicate ambiguous class definitions).

### 10.3 LLM-as-judge cross-check

The same 50 gold turns are also labeled by Claude using this guide as
a system prompt. Cohen's kappa between human-gold and LLM labels is
reported. This validates the guide itself: if the LLM disagrees
systematically with the human on one class, that class definition needs
sharpening before V2.

### 10.4 Acceptance thresholds

- Distant supervision vs gold: Cohen's kappa ≥ 0.45 (this is moderate
  agreement; lower bound is acceptable because distant labels are
  dialogue-level and gold are turn-level).
- LLM-as-judge vs gold: Cohen's kappa ≥ 0.55.
- Below these, we iterate the prompt or guide and report the iteration
  in the methodology section.