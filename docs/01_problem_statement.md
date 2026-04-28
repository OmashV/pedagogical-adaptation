# Problem Statement — Component 2: Failure-Attributed Pedagogical Adaptation

**Author:** [YOUR NAME] ([YOUR ID])
**Component of:** "Towards Artificial General Intelligence through Self-Improving
Multi-Agent Systems for Personalized Learning & Academic Support System"
**Module:** IT4010 Research Project
**Date:** [today's date]

---

## 1. Background

Large Language Models (LLMs) have shown strong performance on solving academic
problems, especially in mathematics. However, solving a problem is not the same
as teaching one. A good tutor does not just produce the correct answer; the tutor
shapes the student's thinking through scaffolded questions, manages cognitive
load, addresses misconceptions, and adjusts the strategy when the student is
struggling.

Recent research has documented this gap clearly. Macina et al. (2023), in the
MathDial paper, show that strong solvers like GPT-3 fail at tutoring because
they generate factually correct but pedagogically poor responses, often revealing
solutions too early. The BEA 2025 Shared Task (Maurya et al., 2025) further
demonstrates that LLM tutor outputs vary widely along pedagogical dimensions
even when the responses are surface-level helpful.

This component focuses on what happens **after** a tutor turn is produced:
detecting when that turn failed pedagogically, attributing the failure to a
specific dimension, and adapting the next strategy within the same conversation.

## 2. Problem statement

Current LLM-based tutoring systems lack a principled mechanism to detect,
attribute, and repair pedagogical failures within a live conversation. Three
specific limitations exist:

1. **Reward signals are not pedagogical.** Standard LLM training optimizes for
   helpfulness, harmlessness, and honesty. A tutor turn that is correct and
   clear can still be pedagogically wrong — too much information, wrong
   scaffolding level, or bypassing productive struggle. There is no widely
   adopted reward model that decomposes pedagogical quality into interpretable
   dimensions.

2. **Failure detection is coarse.** Most systems detect confusion through
   surface signals (the student says "I don't understand", or the answer is
   wrong). This conflates very different failure modes — misconception failures,
   scaffolding-mismatch failures, cognitive-load failures — and prevents
   targeted repair.

3. **In-conversation adaptation is heuristic.** When systems do adapt, the
   adaptation is rule-based rather than grounded in causal attribution or
   principled exploration over a defined strategy space.

Together these limitations mean that LLM tutors miss opportunities to repair
their own teaching strategy mid-conversation, and instead continue with the
same approach that already failed.

## 3. Research objective

To design and evaluate a failure-attributed adaptive tutoring component that:

(i) infers fine-grained learner state from the student's turn,
(ii) scores tutor turns on decomposed pedagogical dimensions via a process
     reward model,
(iii) attributes pedagogical failure to a specific dimension when scoring
      drops, and
(iv) repairs strategy selection within the conversation through bandit-based
     exploration over a typed pedagogical action space.

## 4. Sub-objectives

1. Define a typed taxonomy of pedagogical strategies that the tutor agent
   can be directed to use.
2. Define a decomposed rubric of pedagogical quality dimensions for
   per-turn evaluation.
3. Build and fine-tune a Learner State Inferrer (LSI) module on annotated
   tutoring dialogues.
4. Build a Pedagogical Process Reward Model (Ped-PRM) that scores tutor
   turns along the rubric dimensions, leveraging existing expert
   annotations from the BEA 2025 Shared Task.
5. Implement a Failure Attribution Engine (FAE) that identifies the
   weakest dimension when overall reward drops below threshold.
6. Integrate a strategy selection policy that consumes the attributed
   failure and proposes the next teaching strategy.
7. Evaluate the component on real dialogues from MathDial with intrinsic
   metrics, ablation studies, and (for V2) a human user pilot.

## 5. Scope of this component

This component operates at the **turn level**, **within a single conversation**.
It is invoked after every tutor turn during a live dialogue. It does not perform
across-session learning — that is the responsibility of the Meta-agent
(Component 4). It does not retrieve long-term student history — that is the
responsibility of the Memory component (Component 3). The boundary is
intentional and strict.

## 6. Expected contributions

1. A typed taxonomy of six pedagogical strategies operationalized for
   LLM-based tutoring.
2. A four-dimension rubric for decomposed pedagogical quality evaluation.
3. A trained Learner State Inferrer for confusion-type and
   misconception-flag prediction on MathDial.
4. A Pedagogical Process Reward Model V1 that combines BEA 2025 expert
   annotations with an additional cognitive-load dimension.
5. A basic Failure Attribution Engine over the rubric dimensions.
6. An end-to-end V1 pipeline that demonstrates inference → scoring →
   attribution → strategy directive on real MathDial dialogues.

## 7. Constraints and assumptions

- No synthetic or simulated student data is generated by this project.
  Pedagogical ground truth is sourced from the BEA 2025 Shared Task
  expert annotations on MathDial / Bridge.
- LLM-as-judge is used for automated scoring against rubrics, not for
  generating dialogue content.
- Computation is performed on Google Colab T4 GPU (free tier);
  model size and dataset size are scoped accordingly.

## 8. References

- Macina, J., Daheim, N., Pal Chowdhury, S., Sinha, T., Kapur, M.,
  Gurevych, I., & Sachan, M. (2023). *MathDial: A Dialogue Tutoring
  Dataset with Rich Pedagogical Properties Grounded in Math Reasoning
  Problems.* Findings of EMNLP 2023.
- Maurya, K., et al. (2025). *Pedagogical Evaluation of AI Tutors.*
  BEA 2025 Shared Task. (Verify exact citation when finalizing.)
- Vygotsky, L. S. (1978). *Mind in society: The development of higher
  psychological processes.*
- Wood, D., Bruner, J., & Ross, G. (1976). *The role of tutoring in
  problem solving.*
- Kapur, M. (2008). *Productive failure.*