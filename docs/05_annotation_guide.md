---

## 11. Synthetic data policy

This project distinguishes between primary and secondary evidence.

### Primary (real) evidence — required for all main claims
- Real dialogues: MathDial (human teachers + expert annotations).
- Real distant supervision: LLM classification of real teacher
  confusion descriptions.
- Real human gold labels: 50 turns annotated by the project author.
- Real human pilot study (deferred to V2).

### Secondary (synthetic) support — restricted use
Synthetic data is used only in narrow, transparent roles:

- **Class augmentation:** if a minority class (e.g., procedural) is
  too small for the model to learn, we augment via LLM paraphrasing
  of real turns. Augmentation is applied only to training data;
  validation and test sets are 100% real. Augmentation counts and
  ratios are reported.
- **Prompt calibration:** for the cognitive-load dimension (D4),
  small handcrafted synthetic examples are used as few-shot anchors
  in the LLM-as-judge prompt. These do not appear in any evaluation
  set.
- **Unit test fixtures:** the `tests/` folder uses small synthetic
  edge-case dialogues to validate code behavior. These have no role
  in evaluation.

### Off-limits
- Synthetic dialogues used as if they were real evaluation data.
- Simulated learning outcomes as headline results.
- Synthetic turns added to the gold test set.
- LLM-generated student responses used in pilot studies.

All synthetic-data usage is logged and reported in the methodology
section of the final write-up.