# Component 2: Failure-Attributed Pedagogical Adaptation

Final-year research component for the AGI-towards multi-agent personalized learning system.

## Overview

This component detects pedagogical failures in tutor responses, attributes them
to specific failure dimensions, and adapts the next tutoring strategy within the
same conversation.

## Status

Version 1 (PP1 milestone, 50% complete) — in development.

## Tech stack

- Python 3.11
- PyTorch + Hugging Face Transformers (for the Learner State Inferrer)
- Hugging Face datasets (MathDial)
- Anthropic / OpenAI APIs (for LLM-as-judge in the Pedagogical Process Reward Model)
- Streamlit (for the annotation tool)

## Setup

See `docs/setup.md` (coming soon).

## Folder structure

See `docs/02_architecture.md` (coming soon).
