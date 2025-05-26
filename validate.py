# evaluate_model.py

import os
import json
import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional
import numpy as np  # For metrics and confusion matrix

# --- Import Scikit-learn metrics ---
try:
    from sklearn.metrics import classification_report, confusion_matrix

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print(
        "WARNING: scikit-learn not found. Metrics like classification_report and confusion_matrix will not be available."
    )

    # Define dummy functions if sklearn is not available
    def classification_report(y_true, y_pred, target_names=None, zero_division=0):
        print("sklearn.metrics.classification_report not available.")
        return "Metrics unavailable (sklearn not installed)"

    def confusion_matrix(y_true, y_pred, labels=None):
        print("sklearn.metrics.confusion_matrix not available.")
        return np.array([])  # Return empty array


try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print(
        "WARNING: matplotlib or seaborn not found. Confusion matrix plotting will be disabled."
    )


# --- Import Custom Modules ---
try:
    from preprocessor import Preprocessor
    from code_normalizer import CodeNormalizer
    from text_embedder import TextEmbedder
    from code_embedder import CodeEmbedderGNN
    from concat_embedder import ConcatEmbedder
    from submission_dataset import (
        SubmissionDataset,
    )  # We'll use its _encode_verdict if needed, and structure
    from submission_predictor import SubmissionPredictor
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}. Ensure all component .py files are accessible.")
    exit()

# --- Configuration (Similar to analyze_model_predictions.py) ---
TARGET_LANG = "python"  # Options: "python", "java", "cpp" (lowercase)

SAVE_COMPONENTS_BASE_DIR = "saved_model_components"
LANG_SAVE_DIR = os.path.join(SAVE_COMPONENTS_BASE_DIR, TARGET_LANG)

MODEL_FILE = os.path.join(LANG_SAVE_DIR, f"submission_predictor_{TARGET_LANG}.pth")
GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE = os.path.join(
    SAVE_COMPONENTS_BASE_DIR, "global_text_embedder.joblib"
)
CODE_GNN_VOCAB_FILE = os.path.join(LANG_SAVE_DIR, f"code_gnn_vocab_{TARGET_LANG}.json")

# Dataset CSV paths (USE YOUR VALIDATION OR TEST SET FOR PROPER EVALUATION)
# If you used random_split in train_model.py, you don't have separate CSVs for val/test
# In that case, you'd typically evaluate on the `val_dataset` part.
# For this script to be general, it assumes you might point it to specific evaluation CSVs.
# If you want to evaluate on the validation split from training, you'd need to save that split
# or re-create the split here using the same seed/logic.
# For now, let's assume we use the *same* CSVs as training but would ideally use a held-out test set.
PROBLEM_CSV_EVAL = "data/final_problem_statements.csv"
SUBMISSIONS_CODE_CSV_EVAL_TPL = "data/submissions_{lang_cap}.csv"
SUBMISSION_STATS_CSV_EVAL_TPL = "data/submission_stats_{lang_cap}.csv"

# Model Hyperparameters (MUST match the architecture of the saved model)
CODE_GNN_NODE_VOCAB_SIZE_EVAL = 2000
CODE_GNN_NODE_EMB_DIM_EVAL = 128
CODE_GNN_HIDDEN_DIM_EVAL = 256
CODE_GNN_OUT_DIM_EVAL = 128
CODE_GNN_LAYERS_EVAL = 3
CONCAT_USE_PROJECTION_EVAL = True
CONCAT_PROJECTION_SCALE_EVAL = 0.4
NUM_VERDICT_CLASSES = 7
PREDICTOR_MLP_HIDDEN_DIMS_EVAL = [256, 256, 128, 64]
NUM_VERDICT_CLASSES_EVAL = 7

DEVICE_EVAL = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE_EVAL = 32  # Can be larger for inference than training if memory allows
NUM_WORKERS_EVAL = 2  # For DataLoader
PIN_MEMORY_DATALOADER = False

# For mapping IDs to Verdict Strings (ensure this matches training and SubmissionDataset)
ID_TO_VERDICT_MAP_EVAL = {
    0: "Accepted",
    1: "Wrong Answer",
    2: "Time Limit Exceeded",
    3: "Memory Limit Exceeded",
    4: "Runtime Error",
    5: "Compile Error",
    6: "Presentation Error",
    # -1 is for unknown/unmapped by SubmissionDataset's _encode_verdict
}
# Create a list of target names for classification_report, ordered by ID
TARGET_NAMES_EVAL = [ID_TO_VERDICT_MAP_EVAL[i] for i in range(NUM_VERDICT_CLASSES_EVAL)]


# --- Custom Collate Function (from train_model.py) ---
def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    collated_batch: Dict[str, Any] = {}
    if not batch:
        return collated_batch
    first_item_keys = batch[0].keys()
    for key in first_item_keys:
        if isinstance(batch[0][key], torch.Tensor):
            collated_batch[key] = torch.stack([item[key] for item in batch])
        elif isinstance(batch[0][key], str):
            collated_batch[key] = [item[key] for item in batch]
        elif isinstance(batch[0][key], (int, float)):
            collated_batch[key] = torch.tensor([item[key] for item in batch])
        else:
            collated_batch[key] = [item[key] for item in batch]
    if (
        "lang_str" in collated_batch
        and isinstance(collated_batch["lang_str"], list)
        and collated_batch["lang_str"]
    ):
        collated_batch["lang_str_unified"] = collated_batch["lang_str"][0]
    return collated_batch


def evaluate_model():
    print(f"--- Starting Model Evaluation for Language: {TARGET_LANG.upper()} ---")
    print(f"Using device: {DEVICE_EVAL}")
    if not SKLEARN_AVAILABLE:
        print(
            "ERROR: scikit-learn is not available. Cannot generate detailed metrics. Please install it."
        )
        return

    # 1. Load Components (similar to analyze_model_predictions.py)
    try:
        preprocessor = Preprocessor()
        code_normalizer = CodeNormalizer()
        print("Preprocessor and CodeNormalizer initialized.")

        print(
            f"Loading Global TextEmbedder from: {GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE}"
        )
        if not os.path.exists(GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE):
            print(
                f"ERROR: Global TextEmbedder file NOT FOUND at '{GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE}'."
            )
            return
        tfidf_vectorizer_fitted = joblib.load(GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE)
        global_text_embedder = TextEmbedder(vectorizer_model=tfidf_vectorizer_fitted)
        global_text_embedder._fitted = True
        if not (
            hasattr(global_text_embedder.vectorizer, "vocabulary_")
            and global_text_embedder.vectorizer.vocabulary_
        ):
            print("WARNING: Loaded Global TfidfVectorizer does not appear fitted.")
            global_text_embedder._fitted = False
        if not global_text_embedder._fitted:
            print("ERROR: Global TextEmbedder not properly fitted.")
            return
        print(
            f"Global TextEmbedder loaded. Vocab: {len(global_text_embedder.get_feature_names())}"
        )

        print(f"Loading CodeEmbedderGNN vocab from: {CODE_GNN_VOCAB_FILE}")
        if not os.path.exists(CODE_GNN_VOCAB_FILE):
            print(f"ERROR: CodeGNN vocab file NOT FOUND at '{CODE_GNN_VOCAB_FILE}'.")
            return
        with open(CODE_GNN_VOCAB_FILE, "r") as f:
            code_gnn_vocab_data = json.load(f)
        code_embedder_gnn = CodeEmbedderGNN(
            code_gnn_vocab_data.get("node_vocab_size", CODE_GNN_NODE_VOCAB_SIZE_EVAL),
            CODE_GNN_NODE_EMB_DIM_EVAL,
            CODE_GNN_HIDDEN_DIM_EVAL,
            CODE_GNN_OUT_DIM_EVAL,
            CODE_GNN_LAYERS_EVAL,
        ).to(DEVICE_EVAL)
        code_embedder_gnn.node_type_to_id = code_gnn_vocab_data["node_type_to_id"]
        code_embedder_gnn.next_node_type_id = code_gnn_vocab_data["next_node_type_id"]
        print(f"CodeEmbedderGNN for {TARGET_LANG} initialized and vocab loaded.")

        concat_embedder = ConcatEmbedder(
            code_embedder_gnn,
            global_text_embedder,
            CONCAT_USE_PROJECTION_EVAL,
            CONCAT_PROJECTION_SCALE_EVAL,
        ).to(DEVICE_EVAL)
        print(f"ConcatEmbedder initialized. Final dim: {concat_embedder.final_dim}")

        print(f"Loading SubmissionPredictor model from: {MODEL_FILE}")
        if not os.path.exists(MODEL_FILE):
            print(f"ERROR: Model file NOT FOUND at '{MODEL_FILE}'.")
            return
        model = SubmissionPredictor(
            concat_embedder, NUM_VERDICT_CLASSES_EVAL, PREDICTOR_MLP_HIDDEN_DIMS_EVAL
        ).to(DEVICE_EVAL)
        model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE_EVAL))
        model.eval()  # CRITICAL: Set to evaluation mode
        print("SubmissionPredictor model loaded and set to evaluation mode.")

    except Exception as e:
        print(f"ERROR during component loading: {type(e).__name__} - {e}")
        return

    # 2. Load Evaluation Dataset
    print(f"\n--- Setting up Evaluation Dataset for {TARGET_LANG.upper()} ---")
    lang_cap_for_file = TARGET_LANG.capitalize()
    problem_csv_eval_path = PROBLEM_CSV_EVAL
    submissions_code_csv_eval_path = SUBMISSIONS_CODE_CSV_EVAL_TPL.format(
        lang_cap=lang_cap_for_file
    )
    submission_stats_csv_eval_path = SUBMISSION_STATS_CSV_EVAL_TPL.format(
        lang_cap=lang_cap_for_file
    )

    # Verify actual files exist before proceeding
    for f_path in [
        problem_csv_eval_path,
        submissions_code_csv_eval_path,
        submission_stats_csv_eval_path,
    ]:
        if not os.path.exists(f_path):
            print(
                f"ERROR: Evaluation data file not found: {f_path}. Please check paths."
            )
            return

    try:
        eval_dataset = SubmissionDataset(
            stats_csv_path=submission_stats_csv_eval_path,
            code_csv_path=submissions_code_csv_eval_path,
            problem_csv_path=problem_csv_eval_path,
            preprocessor_instance=preprocessor,
            code_normalizer_instance=code_normalizer,
        )
        if len(eval_dataset) == 0:
            print(f"ERROR: Evaluation dataset for {TARGET_LANG} is empty.")
            return
        print(
            f"Evaluation dataset loaded with {len(eval_dataset)} samples for {TARGET_LANG}."
        )
    except Exception as e:
        print(f"ERROR initializing evaluation SubmissionDataset for {TARGET_LANG}: {e}")
        return

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,  # No shuffle for evaluation
        collate_fn=custom_collate_fn,
        num_workers=NUM_WORKERS_EVAL,
        pin_memory=PIN_MEMORY_DATALOADER,  # From train_model.py config
    )

    # 3. Collect Predictions and True Labels
    print(f"\n--- Collecting predictions for evaluation ---")
    all_true_labels = []
    all_predicted_labels = []
    model.eval()  # Ensure model is in eval mode

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(eval_dataloader):
            codes_batch = batch_data["code_str"]
            stmts_batch = batch_data["statement_str"]
            lang_for_batch = batch_data["lang_str_unified"]  # Should be TARGET_LANG

            true_verdicts_batch = batch_data["verdict_encoded"]  # Already a tensor

            # Ensure all inputs for model are appropriate (model expects lists for code/stmt)
            verdict_logits = model(codes_batch, stmts_batch, lang_for_batch)

            probabilities = torch.softmax(verdict_logits, dim=1)
            predicted_ids_batch = torch.argmax(probabilities, dim=1)

            # Filter out samples with true label -1 (unmapped) before adding to lists for sklearn metrics
            valid_mask = true_verdicts_batch != -1

            all_true_labels.extend(true_verdicts_batch[valid_mask].cpu().numpy())
            all_predicted_labels.extend(predicted_ids_batch[valid_mask].cpu().numpy())

            if (batch_idx + 1) % (max(1, len(eval_dataloader) // 10)) == 0 or (
                batch_idx + 1
            ) == len(eval_dataloader):
                print(f"  Processed batch {batch_idx+1}/{len(eval_dataloader)}")

    if not all_true_labels:
        print(
            "No valid samples found in the evaluation dataset to calculate metrics (all might have been ignore_index)."
        )
        return

    print(f"\n--- Evaluation Metrics for {TARGET_LANG.upper()} ---")
    # 4. Calculate and Print Classification Report
    # Ensure labels used in report are only those present in true/pred and are 0 to N-1
    unique_labels_present = sorted(
        list(set(all_true_labels) | set(all_predicted_labels))
    )
    current_target_names = [
        ID_TO_VERDICT_MAP_EVAL.get(l, f"Unknown_{l}") for l in unique_labels_present
    ]

    report = classification_report(
        all_true_labels,
        all_predicted_labels,
        labels=unique_labels_present,  # Use only labels that are actually present
        target_names=current_target_names,
        zero_division=0,
    )
    print("Classification Report:\n", report)

    # 5. Calculate and Print/Plot Confusion Matrix
    cm_labels = unique_labels_present  # Use the same labels for CM as for report
    cm_target_names = current_target_names

    cm = confusion_matrix(all_true_labels, all_predicted_labels, labels=cm_labels)
    print("\nConfusion Matrix:")
    # Print a simple text version
    cm_df = pd.DataFrame(cm, index=cm_target_names, columns=cm_target_names)
    print(cm_df)

    if PLOTTING_AVAILABLE:
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix for {TARGET_LANG.upper()}")
            plt.ylabel("Actual Verdict")
            plt.xlabel("Predicted Verdict")
            plt.tight_layout()
            # Save the plot
            plot_save_path = os.path.join(
                LANG_SAVE_DIR, f"confusion_matrix_{TARGET_LANG}.png"
            )
            plt.savefig(plot_save_path)
            print(f"\nConfusion matrix plot saved to: {plot_save_path}")
            # plt.show() # Uncomment if running in an environment that supports interactive plotting
            plt.close()
        except Exception as plot_e:
            print(f"Error generating confusion matrix plot: {plot_e}")


if __name__ == "__main__":
    try:
        if not TARGET_LANG.strip():
            raise ValueError(
                "TARGET_LANG configuration is not set at the top of the script."
            )

        # Basic check for essential files
        lang_cap_check = TARGET_LANG.capitalize()
        required_eval_data_files = [
            PROBLEM_CSV_EVAL,
            SUBMISSIONS_CODE_CSV_EVAL_TPL.format(lang_cap=lang_cap_check),
            SUBMISSION_STATS_CSV_EVAL_TPL.format(lang_cap=lang_cap_check),
        ]
        for f_path in required_eval_data_files:
            if not os.path.exists(f_path):
                print(f"ERROR: Required evaluation data file is missing: {f_path}")
                exit()

        required_model_dir = os.path.join(SAVE_COMPONENTS_BASE_DIR, TARGET_LANG)
        if not os.path.isdir(required_model_dir):
            print(
                f"ERROR: Saved model components directory not found: {required_model_dir}"
            )
            exit()
        if (
            not os.path.exists(MODEL_FILE)
            or not os.path.exists(GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE)
            or not os.path.exists(CODE_GNN_VOCAB_FILE)
        ):
            print(
                f"ERROR: One or more critical model component files are missing for {TARGET_LANG}. Check paths."
            )
            exit()

        evaluate_model()
    except ImportError as e_main_imp:
        print(f"Exiting: Import errors: {e_main_imp}")
    except ValueError as ve_main:
        print(f"Configuration error: {ve_main}")
    except Exception as e_main:
        print(
            f"An UNHANDLED error occurred in __main__ execution: {type(e_main).__name__} - {e_main}"
        )
        import traceback

        traceback.print_exc()
