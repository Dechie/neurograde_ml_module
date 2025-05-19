# api_config.py (Updated)

# --- API Configuration ---
# Define the languages this API instance will support and load models for.
# The keys should be the lowercase language identifiers you expect in requests (e.g., "python").
# The values can be used for constructing filenames if they differ slightly (e.g., "Python" for "submissions_Python.csv").
# For simplicity, let's assume the value is the capitalized version for filenames.
API_SUPPORTED_LANGUAGES = {
    "python": "Python",
    "java": "Java",
    "cpp": "Cpp",
    # Add more languages here if you have trained models for them
}

# Base path templates for model components - use {lang_value} for the capitalized lang
# and {lang_key} for the lowercase lang key.
MODEL_FILE_TPL = "submission_predictor_model_{lang_key}.pth"
TEXT_EMBEDDER_VECTORIZER_FILE_TPL = "text_embedder_fitted_{lang_key}.joblib"
CODE_GNN_VOCAB_FILE_TPL = "code_gnn_vocab_{lang_key}.json"

# --- Model Hyperparameters (MUST match the settings used for training the loaded models) ---
# For this multi-language setup, we'll assume these core architectural parameters
# are THE SAME for all language models you are loading.
# If they differ per language, this config section would need to be structured
# as a dictionary keyed by language. For now, keeping it global for simplicity.

# CodeEmbedderGNN params (assumed same architecture for all loaded lang models)
CODE_GNN_NODE_VOCAB_SIZE_API = 100
CODE_GNN_NODE_EMB_DIM_API = 32
CODE_GNN_HIDDEN_DIM_API = 64
CODE_GNN_OUT_DIM_API = 48
CODE_GNN_LAYERS_API = 2

# ConcatEmbedder params (assumed same for all)
CONCAT_USE_PROJECTION_API = False  # Or True, matching training
CONCAT_PROJECTION_SCALE_API = 0.5

# SubmissionPredictor params (assumed same for all)
NUM_VERDICT_CLASSES_API = 7  # Make sure this matches your actual model output
PREDICTOR_MLP_HIDDEN_DIMS_API = [64]

# Device to run inference on
DEVICE_API = "cpu"

# Inverse mapping for verdicts (shared)
ID_TO_VERDICT_MAP = {
    0: "Accepted",
    1: "Wrong Answer",
    2: "Time Limit Exceeded",
    3: "Memory Limit Exceeded",
    4: "Runtime Error",
    5: "Compile Error",
    6: "Presentation Error",
    -1: "Unknown Verdict ID",
}
