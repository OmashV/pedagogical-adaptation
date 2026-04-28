# Pedagogical Strategy Taxonomy

Component 2 selects from this fixed, typed action space when proposing
the next tutoring strategy. Six strategies are defined. The set is designed
to be small enough for a bandit to learn over, and broad enough to cover
the dominant teaching moves observed in the MathDial dataset.

The taxonomy is grounded in three theoretical sources:

- **Scaffolding theory** (Wood, Bruner, & Ross, 1976) — the tutor adjusts
  support to keep the student in productive struggle.
- **Zone of Proximal Development** (Vygotsky, 1978) — the tutor pitches
  at the level just beyond what the student can do alone.
- **MathDial teacher-move taxonomy** (Macina et al., 2023) — Focus,
  Probing, Telling, Generic — which we extend into a finer-grained set.

## The six strategies

### S1 — Socratic question
**Definition:** Ask a focused question that prompts the student to
reason about the next step, without giving the answer.
**When to use:** Student has the prerequisite knowledge but is not
applying it; engagement is moderate.
**Example:** "What operation do you think we should use here, given
that we want a total?"
**Prompt template hint:** "Ask a single Socratic question targeting
the concept: {concept}. Do not provide the answer."

### S2 — Worked example
**Definition:** Walk through a similar but simpler problem step by step.
**When to use:** Student lacks procedural knowledge for this problem
type; multiple Socratic attempts have failed.
**Example:** "Let's try a simpler one first. If a box has 3 apples
and we add 2 more, we count: 3, then 4, then 5. Now back to your
problem..."
**Prompt template hint:** "Walk through a simpler analogous problem
involving {concept}, step by step. Then return to the student's problem."

### S3 — Analogy
**Definition:** Map the abstract concept to a concrete familiar domain.
**When to use:** Student shows abstract-reasoning difficulty; concept is
unfamiliar in its current framing.
**Example:** "Think of multiplication like rows in a garden — 3 rows
of 4 plants is the same as 3 × 4."
**Prompt template hint:** "Explain {concept} using an analogy from
everyday life that the student can relate to."

### S4 — Decomposition
**Definition:** Break the problem into smaller named sub-steps, ask
the student to do one sub-step at a time.
**When to use:** Cognitive load is high; the problem has multiple steps
and the student is overwhelmed.
**Example:** "Let's break this into two parts. First, can you tell me
how many apples Sam has? We'll do the rest after that."
**Prompt template hint:** "Decompose the problem into 2–3 named
sub-steps. Ask only about the first sub-step."

### S5 — Concrete instantiation
**Definition:** Replace abstract variables with specific small numbers
or objects.
**When to use:** Student is stuck on an abstract or symbolic
representation; modality mismatch.
**Example:** "Forget the variable for a moment. Imagine there are
exactly 5 cookies and 2 friends..."
**Prompt template hint:** "Restate the problem with concrete small
numbers and tangible objects."

### S6 — Hint laddering
**Definition:** Provide a graduated hint — first abstract, then
specific, then more specific — only revealing more if the student
remains stuck.
**When to use:** Student has tried and failed once; needs a small
nudge, not a full explanation.
**Example:** "Hint: think about what 'total' means. Need more? It's
related to addition. Still stuck? Try adding the two numbers in the
problem."
**Prompt template hint:** "Provide a three-tier hint: tier 1 abstract,
tier 2 specific, tier 3 very specific. Show only tier 1 unless asked."

## Coverage map to MathDial moves

| MathDial move | Maps to |
|---|---|
| Focus (guiding task progress) | S4 (Decomposition), partially S1 (Socratic) |
| Probing (encouraging conceptual exploration) | S1 (Socratic), partially S3 (Analogy) |
| Telling (providing help when stuck) | S2 (Worked example), S6 (Hint laddering) |
| Generic (greeting, conversational) | Not part of pedagogical action space |

Generic moves are excluded from the action space; they are handled
by the Tutor Agent's default conversational behavior.

## Why six and not more

Bandit algorithms scale poorly with action-space size for short
in-session horizons. Six strategies give enough coverage of the
MathDial move taxonomy while keeping the bandit posterior tractable
within a typical 8–15 turn dialogue.

A seventh strategy ("retrieval practice" — quick recall of a previously
mastered concept) is reserved for V2 and is not in scope for PP1.