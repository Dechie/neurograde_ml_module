# analyze_model_predictions.py

import os
import json
import joblib
import pandas as pd
import torch
from torch.utils.data import (
    DataLoader,
)  # For iterating if needed, though direct iteration is also fine for this task
from typing import Dict, List, Any, Optional

# --- Import Custom Modules ---
try:
    from preprocessor import Preprocessor
    from code_normalizer import CodeNormalizer
    from text_embedder import TextEmbedder
    from code_embedder import CodeEmbedderGNN
    from concat_embedder import ConcatEmbedder
    from submission_dataset import (
        SubmissionDataset,
    )  # We'll use its __getitem__ and _encode_verdict
    from submission_predictor import SubmissionPredictor
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}. Ensure all component .py files are accessible.")
    exit()

# --- Configuration ---
# Specify the language of the pre-trained model and dataset to analyze
TARGET_LANG = "python"  # Options: "python", "java", "cpp" (lowercase)

# Paths to the saved model components (these should match how train_model.py saves them)
SAVE_COMPONENTS_BASE_DIR = "saved_model_components"  # Base directory
LANG_SAVE_DIR = os.path.join(SAVE_COMPONENTS_BASE_DIR, TARGET_LANG)

MODEL_FILE = os.path.join(LANG_SAVE_DIR, f"submission_predictor_{TARGET_LANG}.pth")
# Assuming a globally fitted TextEmbedder was used for all languages during training
# and saved once. If it was saved per language, adjust path.
GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE = os.path.join(
    SAVE_COMPONENTS_BASE_DIR, "global_text_embedder.joblib"
)
# Or if TextEmbedder was saved per language:
# TEXT_EMBEDDER_VECTORIZER_FILE = os.path.join(LANG_SAVE_DIR, f"text_embedder_{TARGET_LANG}.joblib")

CODE_GNN_VOCAB_FILE = os.path.join(LANG_SAVE_DIR, f"code_gnn_vocab_{TARGET_LANG}.json")

# Dataset CSV paths (these should match your actual data structure)
PROBLEM_CSV = "data/final_problem_statements.csv"
SUBMISSIONS_CODE_CSV_TPL = "data/submissions_{lang_cap}.csv"
SUBMISSION_STATS_CSV_TPL = "data/submission_stats_{lang_cap}.csv"

# Model Hyperparameters (MUST match the architecture of the saved model)
# TextEmbedder params used when GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE was created
# Not strictly needed if TextEmbedder just loads the vectorizer_model, but good for reference.
# TEXT_EMB_MAX_FEATURES_ANALYSIS = 5000

# CodeEmbedderGNN params
CODE_GNN_NODE_VOCAB_SIZE_ANALYSIS = 2000  # From training config
CODE_GNN_NODE_EMB_DIM_ANALYSIS = 64
CODE_GNN_HIDDEN_DIM_ANALYSIS = 128
CODE_GNN_OUT_DIM_ANALYSIS = 64
CODE_GNN_LAYERS_ANALYSIS = 2

# ConcatEmbedder params
CONCAT_USE_PROJECTION_ANALYSIS = True  # Match training
CONCAT_PROJECTION_SCALE_ANALYSIS = 0.5

# SubmissionPredictor params
NUM_VERDICT_CLASSES_ANALYSIS = 7
PREDICTOR_MLP_HIDDEN_DIMS_ANALYSIS = [128, 64]

DEVICE_ANALYSIS = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SAMPLES_TO_COLLECT = 100  # Collect up to this many correct and incorrect samples

# Output file names
CORRECT_SAMPLES_CSV = f"data/analysis_correct_{TARGET_LANG}.csv"
INCORRECT_SAMPLES_CSV = f"data/analysis_incorrect_{TARGET_LANG}.csv"

# For mapping predicted ID back to string (ensure this matches training)
ID_TO_VERDICT_MAP_ANALYSIS = {
    0: "Accepted",
    1: "Wrong Answer",
    2: "Time Limit Exceeded",
    3: "Memory Limit Exceeded",
    4: "Runtime Error",
    5: "Compile Error",
    6: "Presentation Error",
    -1: "Unknown Verdict ID",
}


# --- Main Analysis Script ---
def analyze_predictions():
    print(f"--- Starting Prediction Analysis for Language: {TARGET_LANG.upper()} ---")
    print(f"Using device: {DEVICE_ANALYSIS}")

    # 1. Load Preprocessor and CodeNormalizer (stateless)
    try:
        preprocessor = Preprocessor()
        code_normalizer = CodeNormalizer()
        print("Preprocessor and CodeNormalizer initialized.")
    except Exception as e:
        print(f"Error initializing Preprocessor/CodeNormalizer: {e}")
        return

    # 2. Load Globally Fitted TextEmbedder
    print(f"Loading Global TextEmbedder from: {GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE}")
    global_text_embedder: Optional[TextEmbedder] = None
    try:
        if not os.path.exists(GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE):
            print(
                f"ERROR: Global TextEmbedder vectorizer file NOT FOUND at '{GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE}'. Cannot proceed."
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
        print(
            f"Global TextEmbedder loaded. Fitted: {global_text_embedder._fitted}, Vocab: {len(global_text_embedder.get_feature_names()) if global_text_embedder._fitted else 'N/A'}"
        )
        if not global_text_embedder._fitted:
            print("ERROR: Global TextEmbedder is not properly fitted. Cannot proceed.")
            return
    except Exception as e:
        print(f"ERROR loading Global TextEmbedder: {e}")
        return

    # 3. Load CodeEmbedderGNN (with vocab)
    print(f"Loading CodeEmbedderGNN vocab from: {CODE_GNN_VOCAB_FILE}")
    code_embedder_gnn: Optional[CodeEmbedderGNN] = None
    try:
        if not os.path.exists(CODE_GNN_VOCAB_FILE):
            print(
                f"ERROR: CodeGNN vocab file NOT FOUND at '{CODE_GNN_VOCAB_FILE}'. Cannot proceed."
            )
            return
        with open(CODE_GNN_VOCAB_FILE, "r") as f:
            code_gnn_vocab_data = json.load(f)

        code_embedder_gnn = CodeEmbedderGNN(
            node_vocab_size=code_gnn_vocab_data.get(
                "node_vocab_size", CODE_GNN_NODE_VOCAB_SIZE_ANALYSIS
            ),
            node_embedding_dim=CODE_GNN_NODE_EMB_DIM_ANALYSIS,
            hidden_gnn_dim=CODE_GNN_HIDDEN_DIM_ANALYSIS,
            out_graph_embedding_dim=CODE_GNN_OUT_DIM_ANALYSIS,
            num_gnn_layers=CODE_GNN_LAYERS_ANALYSIS,
        ).to(DEVICE_ANALYSIS)
        code_embedder_gnn.node_type_to_id = code_gnn_vocab_data["node_type_to_id"]
        code_embedder_gnn.next_node_type_id = code_gnn_vocab_data["next_node_type_id"]
        print(f"CodeEmbedderGNN for {TARGET_LANG} initialized and vocab loaded.")
    except Exception as e:
        print(f"ERROR loading CodeEmbedderGNN for {TARGET_LANG}: {e}")
        return

    # 4. Load ConcatEmbedder
    print("Initializing ConcatEmbedder...")
    concat_embedder: Optional[ConcatEmbedder] = None
    try:
        concat_embedder = ConcatEmbedder(
            code_embedder=code_embedder_gnn,
            text_embedder=global_text_embedder,
            use_projection=CONCAT_USE_PROJECTION_ANALYSIS,
            projection_dim_scale_factor=CONCAT_PROJECTION_SCALE_ANALYSIS,
        ).to(DEVICE_ANALYSIS)
        print(f"ConcatEmbedder initialized. Final dim: {concat_embedder.final_dim}")
    except Exception as e:
        print(f"ERROR initializing ConcatEmbedder: {e}")
        return

    # 5. Load SubmissionPredictor Model
    print(f"Loading SubmissionPredictor model from: {MODEL_FILE}")
    model: Optional[SubmissionPredictor] = None
    try:
        if not os.path.exists(MODEL_FILE):
            print(f"ERROR: Model file NOT FOUND at '{MODEL_FILE}'. Cannot proceed.")
            return
        model = SubmissionPredictor(
            concat_embedder=concat_embedder,
            num_verdict_classes=NUM_VERDICT_CLASSES_ANALYSIS,
            mlp_hidden_dims=PREDICTOR_MLP_HIDDEN_DIMS_ANALYSIS,
        ).to(DEVICE_ANALYSIS)
        model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE_ANALYSIS))
        model.eval()  # Set to evaluation mode
        print("SubmissionPredictor model loaded and set to evaluation mode.")
    except Exception as e:
        print(f"ERROR loading SubmissionPredictor model: {e}")
        return

    # --- All components should be loaded now ---
    if not all(
        [
            preprocessor,
            code_normalizer,
            global_text_embedder,
            code_embedder_gnn,
            concat_embedder,
            model,
        ]
    ):
        print(
            "CRITICAL: One or more essential components failed to initialize. Aborting analysis."
        )
        return

    # 6. Load the Dataset
    print(f"\n--- Setting up SubmissionDataset for {TARGET_LANG.upper()} ---")
    lang_cap_for_file = TARGET_LANG.capitalize()
    problem_csv_path = PROBLEM_CSV
    submissions_code_csv_path = SUBMISSIONS_CODE_CSV_TPL.format(
        lang_cap=lang_cap_for_file
    )
    submission_stats_csv_path = SUBMISSION_STATS_CSV_TPL.format(
        lang_cap=lang_cap_for_file
    )

    try:
        dataset = SubmissionDataset(
            stats_csv_path=submission_stats_csv_path,
            code_csv_path=submissions_code_csv_path,
            problem_csv_path=problem_csv_path,
            preprocessor_instance=preprocessor,
            code_normalizer_instance=code_normalizer,
            # Note: SubmissionDataset no longer takes concat_embedder directly
        )
        if len(dataset) == 0:
            print(
                f"ERROR: Dataset for {TARGET_LANG} is empty. Cannot perform analysis."
            )
            return
        print(f"Dataset loaded with {len(dataset)} samples for {TARGET_LANG}.")
    except Exception as e:
        print(f"ERROR initializing SubmissionDataset for {TARGET_LANG}: {e}")
        return

    # 7. Iterate through dataset and collect samples
    print(
        f"\n--- Analyzing predictions (collecting up to {MAX_SAMPLES_TO_COLLECT} correct/incorrect samples) ---"
    )
    correctly_predicted_samples = []
    incorrectly_predicted_samples = []

    # For faster iteration if dataset is huge, consider DataLoader or iterating a subset
    # For now, iterate directly over the dataset, can be slow for very large ones

    # Consider using a DataLoader for batching if iterating the whole dataset is too slow
    # analysis_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    # For item-by-item processing as in your __getitem__ for SubmissionDataset:

    samples_processed = 0
    for i in range(len(dataset)):
        if (
            len(correctly_predicted_samples) >= MAX_SAMPLES_TO_COLLECT
            and len(incorrectly_predicted_samples) >= MAX_SAMPLES_TO_COLLECT
        ):
            print("Collected enough samples for both categories.")
            break  # Stop if we have enough of both

        try:
            item_data = dataset[i]  # This is a dictionary from your SubmissionDataset
        except IndexError:
            print(f"Warning: Index {i} out of bounds for dataset. Stopping.")
            break
        except Exception as e:
            print(f"Warning: Error getting item {i} from dataset: {e}. Skipping.")
            continue

        code_str = item_data["code_str"]
        statement_str = item_data["statement_str"]
        lang_str = item_data["lang_str"]  # Should match TARGET_LANG
        true_verdict_encoded = item_data["verdict_encoded"].item()  # Get scalar value
        true_verdict_raw = item_data["verdict_raw_str"]

        if lang_str.lower() != TARGET_LANG.lower():  # Safety check
            print(
                f"Warning: Sample language '{lang_str}' does not match TARGET_LANG '{TARGET_LANG}'. Skipping sample {i}."
            )
            continue

        if true_verdict_encoded == -1:  # Skip unmapped verdicts from dataset
            # print(f"Skipping sample {i} with unmapped true verdict: {true_verdict_raw}")
            continue

        with torch.no_grad():
            # Model's forward pass expects lists
            verdict_logits = model(
                code_list=[code_str], text_list=[statement_str], lang=lang_str
            )

        probabilities = torch.softmax(verdict_logits, dim=1).squeeze()
        predicted_id = torch.argmax(probabilities).item()
        predicted_verdict_raw = ID_TO_VERDICT_MAP_ANALYSIS.get(
            predicted_id, "Unknown_Predicted"
        )

        # Prepare data for saving (original data + prediction)
        sample_info_for_csv = {
            "problem_id": dataset.df.iloc[i].get(
                "problem_id", "N/A"
            ),  # Get original problem_id if possible
            "submission_id": dataset.df.iloc[i].get("submission_id", "N/A"),
            "language": lang_str,
            "code": code_str,  # Normalized code
            "statement_full": (
                preprocessor.preprocess_text(  # Re-create full raw statement for readability if needed
                    dataset.df.iloc[i].get("statement", "")
                    + "\nInput: "
                    + dataset.df.iloc[i].get("input_spec", "")
                    + "\nOutput: "
                    + dataset.df.iloc[i].get("output_spec", "")
                )
                if hasattr(dataset, "df")
                else statement_str
            ),  # Fallback
            "true_verdict_encoded": true_verdict_encoded,
            "true_verdict_string": true_verdict_raw,
            "predicted_verdict_id": predicted_id,
            "predicted_verdict_string": predicted_verdict_raw,
        }
        # Add individual probabilities
        for class_idx, prob_val in enumerate(probabilities):
            class_name = ID_TO_VERDICT_MAP_ANALYSIS.get(
                class_idx, f"ProbClass{class_idx}"
            )
            sample_info_for_csv[f"prob_{class_name.replace(' ', '_')}"] = (
                prob_val.item()
            )

        if predicted_id == true_verdict_encoded:
            if len(correctly_predicted_samples) < MAX_SAMPLES_TO_COLLECT:
                correctly_predicted_samples.append(sample_info_for_csv)
        else:
            if len(incorrectly_predicted_samples) < MAX_SAMPLES_TO_COLLECT:
                incorrectly_predicted_samples.append(sample_info_for_csv)

        samples_processed += 1
        if samples_processed % 200 == 0:  # Log progress
            print(
                f"  Processed {samples_processed}/{len(dataset)} samples. "
                f"Collected: Correct={len(correctly_predicted_samples)}, Incorrect={len(incorrectly_predicted_samples)}"
            )

    print(f"\n--- Analysis Complete ---")
    print(f"Total samples processed: {samples_processed}")
    print(f"Correctly predicted samples collected: {len(correctly_predicted_samples)}")
    print(
        f"Incorrectly predicted samples collected: {len(incorrectly_predicted_samples)}"
    )

    # 8. Save to CSV
    if correctly_predicted_samples:
        df_correct = pd.DataFrame(correctly_predicted_samples)
        df_correct.to_csv(CORRECT_SAMPLES_CSV, index=False)
        print(f"Saved correctly predicted samples to: {CORRECT_SAMPLES_CSV}")

    if incorrectly_predicted_samples:
        df_incorrect = pd.DataFrame(incorrectly_predicted_samples)
        df_incorrect.to_csv(INCORRECT_SAMPLES_CSV, index=False)
        print(f"Saved incorrectly predicted samples to: {INCORRECT_SAMPLES_CSV}")


if __name__ == "__main__":
    try:
        if not TARGET_LANG.strip():
            raise ValueError(
                "TARGET_LANG configuration is not set at the top of the script."
            )
        # Basic check for essential files based on TARGET_LANG
        lang_cap_check = TARGET_LANG.capitalize()
        required_data_files = [
            PROBLEM_CSV,
            SUBMISSIONS_CODE_CSV_TPL.format(lang_cap=lang_cap_check),
            SUBMISSION_STATS_CSV_TPL.format(lang_cap=lang_cap_check),
        ]
        for f_path in required_data_files:
            if not os.path.exists(f_path):
                print(f"ERROR: Required data file for analysis is missing: {f_path}")
                print("Please ensure paths in the configuration section are correct.")
                exit()

        required_model_files_dir = os.path.join(SAVE_COMPONENTS_BASE_DIR, TARGET_LANG)
        if not os.path.isdir(required_model_files_dir):
            print(
                f"ERROR: Saved model components directory not found: {required_model_files_dir}"
            )
            exit()

        analyze_predictions()
    except ImportError as e_main_imp:
        print(f"Exiting due to critical top-level import errors: {e_main_imp}")
    except ValueError as ve_main:
        print(f"Configuration error: {ve_main}")
    except Exception as e_main:
        print(
            f"An UNHANDLED error occurred in __main__ execution: {type(e_main).__name__} - {e_main}"
        )
        import traceback

        traceback.print_exc()
