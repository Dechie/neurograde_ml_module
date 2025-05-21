# api_config.py (Updated for Global TextEmbedder)

import os

# --- API Configuration ---
API_SUPPORTED_LANGUAGES = {
    "python": "Python",
    "java": "Java",
    "cpp": "Cpp",
}

# Base directory where language-specific model component folders are located
SAVED_COMPONENTS_BASE_DIR_API = (
    "saved_model_components"  # e.g., "saved_model_components/"
)

# --- Filename templates WITHIN each language's subdirectory ---
MODEL_FILENAME_TPL = "submission_predictor_{lang_key}.pth"
CODE_GNN_VOCAB_FILENAME_TPL = "code_gnn_vocab_{lang_key}.json"

# --- Path for the GLOBALLY FITTED TextEmbedder's TfidfVectorizer ---
GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE = os.path.join(
    SAVED_COMPONENTS_BASE_DIR_API, "global_text_embedder.joblib"
)

# --- Model Hyperparameters (MUST match training for loaded models) ---
# Architectural parameters for CodeEmbedderGNN, ConcatEmbedder, SubmissionPredictor
# Assumed same architecture for all loaded language models.
CODE_GNN_NODE_VOCAB_SIZE_API = 2000
CODE_GNN_NODE_EMB_DIM_API = 64
CODE_GNN_HIDDEN_DIM_API = 128
CODE_GNN_OUT_DIM_API = 64
CODE_GNN_LAYERS_API = 2

CONCAT_USE_PROJECTION_API = True
CONCAT_PROJECTION_SCALE_API = 0.5

NUM_VERDICT_CLASSES_API = 7
PREDICTOR_MLP_HIDDEN_DIMS_API = [128, 64]

# TextEmbedder specific hyperparams used when the GLOBAL vectorizer was created (if TextEmbedder class needs them for init)
# However, if TextEmbedder just takes a vectorizer_model, only the loaded vectorizer is needed.
# TEXT_EMB_MAX_FEATURES_API = 5000 # This would have been used to fit the global vectorizer.
# Not strictly needed for loading if the vectorizer itself is loaded.

DEVICE_API = "cpu"

# api_config.py (relevant parts)

NUM_VERDICT_CLASSES_API = 7

ID_TO_VERDICT_MAP = {
    0: "Accepted",
    1: "Wrong Answer",
    2: "Time Limit Exceeded",
    3: "Memory Limit Exceeded",
    4: "Runtime Error",
    5: "Compile Error",
    6: "Presentation Error",
    -1: "Unknown Verdict ID",  # Should not happen with a softmax output over known classes
}

# For easy lookup from string to ID (used in score calculation)
VERDICT_TO_ID_MAP_API = {v: k for k, v in ID_TO_VERDICT_MAP.items() if k != -1}

ENABLE_LLM_REVIEW = True  # Set to False to disable review generation
LLM_REVIEW_PROMPT_TEMPLATE = """
Please analyze the following prediction regarding a student's {language} code submission to a problem assigned by their instructor.

The predicted primary verdict for this submission is: "{predicted_verdict_string}".

The model has also provided the following probability distribution across all possible verdicts:
{verdict_probabilities_str}

Additionally, the submission received:
- A **Correctness Score** of {correctness_score:.2f} (range: 0.0 to 1.0; higher values indicate greater alignment with expected behavior)
- An **Efficiency Score** of {efficiency_score:.2f} (range: -1.0 to 1.0; higher values indicate better performance and resource usage)

Based on this analysis, provide a brief and informative review (2–3 sentences) for the instructor. Your response should:
- Acknowledge success if the predicted verdict is "Accepted" and optionally highlight the strong scores.
- If the verdict indicates a failure (e.g., "Wrong Answer", "Compile Error", "Runtime Error"), identify the general area of concern and suggest likely causes or next steps the student could take.
- If the verdict is "Time Limit Exceeded" or "Memory Limit Exceeded", comment on possible inefficiencies or excessive resource usage in the implementation.
- If the verdict is "Presentation Error", recommend checking formatting or output conventions.

The tone should remain professional, objective, and constructive, providing insight that may assist the instructor in understanding the submission quality.
"""
