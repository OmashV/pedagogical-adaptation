"""
Central configuration for Component 2.

All paths, constants, and hyperparameters live here. Other modules
import from this file. Do NOT hardcode paths anywhere else.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file at project root
load_dotenv()

# ---- Project paths ----
# This file lives at <PROJECT_ROOT>/src/config.py, so the project
# root is two levels up from this file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATED_DIR = DATA_DIR / "annotated"
SCHEMAS_DIR = DATA_DIR / "schemas"

MODELS_DIR = PROJECT_ROOT / "models"
LSI_MODEL_DIR = MODELS_DIR / "lsi"

RESULTS_DIR = PROJECT_ROOT / "results"
LSI_EVAL_DIR = RESULTS_DIR / "lsi_eval"
PED_PRM_EVAL_DIR = RESULTS_DIR / "ped_prm_eval"
DEMO_RUNS_DIR = RESULTS_DIR / "demo_runs"

DOCS_DIR = PROJECT_ROOT / "docs"

# ---- API keys (read from .env) ----
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---- LLM settings (Groq) ----
GROQ_MODEL = "llama-3.1-8b-instant"  # smaller, much higher effective throughput
GROQ_RPM_LIMIT = 12  # well under free-tier RPM and TPM limits
GROQ_REQUEST_DELAY_SEC = 5.0  # 60/12 = 5.0 seconds between calls

# ---- Dataset settings ----
MATHDIAL_HF_ID = "eth-nlped/mathdial"
MATHDIAL_LOCAL_RAW_FILE = RAW_DIR / "mathdial" / "mathdial_raw.parquet"
MATHDIAL_PROCESSED_TURNS_FILE = PROCESSED_DIR / "mathdial_turns.parquet"

# ---- Distant supervision sampling ----
DISTANT_SAMPLE_PLAN = {
    "Yes": 400,
    "Yes, but I had to reveal the answer": 400,
    "No": 200,
}
DISTANT_RANDOM_SEED = 42

# ---- LSI labels (V1 scope) ----
CONFUSION_TYPES = ["none", "lexical", "conceptual", "procedural"]
MISCONCEPTION_FLAG_VALUES = [0, 1]

# ---- Strategy taxonomy (V1 scope) ----
STRATEGIES = ["socratic", "worked_example", "analogy",
              "decomposition", "concrete_instantiation", "hint_laddering"]

# ---- Rubric dimensions (V1 scope) ----
RUBRIC_DIMENSIONS = ["misconception_address", "scaffolding_fit",
                     "zpd_fit", "cognitive_load"]

# ---- Failure threshold ----
FAILURE_AGGREGATE_THRESHOLD = 0.5
FAILURE_DIMENSION_FLOOR = 1  # any dimension scoring 1 triggers failure

# ---- LSI training defaults (used Day 6+) ----
LSI_BASE_MODEL = "microsoft/deberta-v3-base"
LSI_MAX_LENGTH = 256
LSI_BATCH_SIZE = 8
LSI_LEARNING_RATE = 2e-5
LSI_NUM_EPOCHS = 4
LSI_RANDOM_SEED = 42


def ensure_dirs():
    """Create any missing project directories. Safe to call repeatedly."""
    for d in [RAW_DIR / "mathdial", PROCESSED_DIR, ANNOTATED_DIR,
              SCHEMAS_DIR, LSI_MODEL_DIR, LSI_EVAL_DIR,
              PED_PRM_EVAL_DIR, DEMO_RUNS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("HF_TOKEN loaded:", "yes" if HF_TOKEN else "NO")
    ensure_dirs()
    print("All directories ensured.")