# Pedagogical Quality Rubric (Ped-PRM Dimensions)

Component 2 scores every tutor turn along four pedagogical dimensions.
Three dimensions reuse expert annotations from the BEA 2025 Shared
Task on MathDial / Bridge. The fourth dimension (cognitive load) is
not covered by BEA and is scored by an LLM-as-judge against the rubric
defined here, validated against a small manually-scored set.

For BEA-aligned dimensions we adopt their 3-level scheme
(Yes / To some extent / No, mapped to scores 3 / 2 / 1) for V1,
and treat the cognitive-load dimension on the same 3-level scale
for consistency.

## D1 — Misconception-Address

**What it measures:** Does the tutor turn correctly identify and
address the student's underlying error or misconception?

This dimension is a combination of BEA's "Mistake Identification"
and "Mistake Location" criteria.

| Score | Anchor |
|---|---|
| 3 (Yes) | The turn identifies the student's specific error and addresses it directly. |
| 2 (Partial) | The turn references that something is wrong but does not pinpoint the specific error. |
| 1 (No) | The turn proceeds as if there were no error, or addresses a different error than the student made. |

## D2 — Scaffolding-Fit (Providing Guidance)

**What it measures:** Does the turn provide the right level of
guidance — enough to help, not so much that it gives away the answer?

This dimension corresponds to BEA's "Providing Guidance" criterion.

| Score | Anchor |
|---|---|
| 3 (Yes) | The turn nudges the student forward without revealing the answer; preserves productive struggle. |
| 2 (Partial) | The turn helps somewhat but either leaks too much of the solution or gives too thin a nudge. |
| 1 (No) | The turn either gives the answer outright (over-telling) or provides no actionable guidance. |

## D3 — ZPD-Fit (Actionability)

**What it measures:** Is the turn's request matched to what the
student can plausibly do next, given the student's demonstrated
state in the dialogue?

This corresponds to BEA's actionability dimension (verify exact
naming when finalizing citations).

| Score | Anchor |
|---|---|
| 3 (Yes) | What the turn asks the student to do is achievable from the student's current position with appropriate effort. |
| 2 (Partial) | What is asked is partially achievable but stretches beyond the visible state, or undershoots and is trivial. |
| 1 (No) | What is asked is far beyond the student's current ability, or far below it (no learning value). |

## D4 — Cognitive-Load (our addition)

**What it measures:** Does the turn manage the amount of new
information so the student can process it without being overwhelmed?

This dimension is **not** covered by BEA. It is scored by LLM-as-judge
against this rubric and validated against ~50 manually-scored turns.

| Score | Anchor |
|---|---|
| 3 (Yes) | The turn introduces at most one new idea; it is concise and focused. |
| 2 (Partial) | The turn introduces 2 ideas, or is verbose but stays on a single concept. |
| 1 (No) | The turn introduces 3+ ideas at once, or is excessively long, or jumps between unrelated concepts. |

## Aggregation

The aggregate Ped-PRM score for a turn is the unweighted mean of
the four dimensions, scaled to [0, 1]:

  aggregate = (D1 + D2 + D3 + D4 - 4) / 8

A turn is flagged as a **failure candidate** when:
  aggregate < 0.5 OR any single dimension == 1

The Failure Attribution Engine then identifies the lowest-scoring
dimension(s) for that turn.

## Why three levels and not five

For V1 we adopt BEA's 3-level scheme directly so we can use their
expert annotations without re-mapping. V2 will explore expanding to a
5-level scale where finer-grained discrimination is needed and where
re-annotation budget allows.

## Inter-rater reliability targets for our LLM-as-judge

For D4 (cognitive-load), we report Spearman correlation between the
LLM-as-judge score and our manual rubric score on a 50-turn validation
set. Acceptance threshold for V1: ρ ≥ 0.5. Below this, we iterate the
rubric prompt before integrating into the pipeline.