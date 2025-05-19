# train_model.py

import os
import shutil
import pandas as pd
import json  # For saving CodeGNN vocab
import joblib  # For saving TextEmbedder vectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Any

# --- Import all your custom modules ---
try:
    from preprocessor import Preprocessor
    from code_normalizer import CodeNormalizer
    from text_embedder import TextEmbedder
    from code_embedder import CodeEmbedderGNN
    from concat_embedder import ConcatEmbedder
    from submission_dataset import SubmissionDataset  # Updated version
    from submission_predictor import SubmissionPredictor  # No changes needed here
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR in train_model.py: {e}")
    exit()

# --- Configuration Section (same as before) ---
USE_DUMMY_DATA = True
DUMMY_DATA_DIR = "temp_training_dummy_data_v2"
DUMMY_PROBLEM_CSV = os.path.join(DUMMY_DATA_DIR, "dummy_problem_statements.csv")
DUMMY_SUBMISSIONS_CODE_CSV_TPL = os.path.join(
    DUMMY_DATA_DIR, "dummy_submissions_{lang}.csv"
)
DUMMY_SUBMISSION_STATS_CSV_TPL = os.path.join(
    DUMMY_DATA_DIR, "dummy_submission_stats_{lang}.csv"
)

ACTUAL_PROBLEM_CSV = "path/to/your/final_problem_statements.csv"
ACTUAL_SUBMISSIONS_CODE_CSV_TPL = "path/to/your/submissions_{lang}.csv"
ACTUAL_SUBMISSION_STATS_CSV_TPL = "path/to/your/submission_stats_{lang}.csv"

TARGET_LANG = "cpp"
TEXT_EMB_MAX_FEATURES = 100
CODE_GNN_NODE_VOCAB_SIZE = 100
CODE_GNN_NODE_EMB_DIM = 32
CODE_GNN_HIDDEN_DIM = 64
CODE_GNN_OUT_DIM = 48
CODE_GNN_LAYERS = 2
CONCAT_USE_PROJECTION = False  # Set to False for simplicity in this version
CONCAT_PROJECTION_SCALE = 0.5
NUM_VERDICT_CLASSES = 7
PREDICTOR_MLP_HIDDEN_DIMS = [64]  # Simpler MLP for demo
LEARNING_RATE = 1e-4
BATCH_SIZE = 4  # Smaller for dummy data
NUM_EPOCHS = 3  # Fewer epochs for quick demo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = f"submission_predictor_model_{TARGET_LANG}.pth"
TEXT_EMBEDDER_SAVE_PATH = (
    f"text_embedder_fitted_{TARGET_LANG}.joblib"  # Changed to .joblib
)
CODE_GNN_VOCAB_SAVE_PATH = f"code_gnn_vocab_{TARGET_LANG}.json"


# --- Helper Function to Create Dummy Data (same as before) ---
def create_dummy_data(lang_for_demo: str):
    print(f"Creating dummy data in {DUMMY_DATA_DIR} for language: {lang_for_demo}...")
    os.makedirs(DUMMY_DATA_DIR, exist_ok=True)
    problem_data = {
        "problem_id": [f"p{i}" for i in range(1, 6)],
        "statement": [f"Statement p{i}." for i in range(1, 6)],
        "input_spec": [f"Input p{i}." for i in range(1, 6)],
        "output_spec": [f"Output p{i}." for i in range(1, 6)],
    }
    pd.DataFrame(problem_data).to_csv(DUMMY_PROBLEM_CSV, index=False)
    submissions_code_data = {
        "problem_id": [f"p{(i%3)+1}" for i in range(10)],  # p1,p2,p3
        "submission_file": [f"s{i}.{lang_for_demo}" for i in range(10)],
        "code": [f"def solve{i}(): return {i} # {lang_for_demo}" for i in range(10)],
    }
    pd.DataFrame(submissions_code_data).to_csv(
        DUMMY_SUBMISSIONS_CODE_CSV_TPL.format(lang=lang_for_demo.capitalize()),
        index=False,
    )
    verdicts = [
        "Accepted",
        "Wrong Answer",
        "Time Limit Exceeded",
        "Memory Limit Exceeded",
        "Runtime Error",
        "Compile Error",
        "Presentation Error",
    ]
    submission_stats_data = {
        "problem_id": [f"p{(i%3)+1}" for i in range(10)],
        "submission_id": [f"s{i}" for i in range(10)],
        "language": [lang_for_demo.lower()] * 10,
        "verdict": [verdicts[i % len(verdicts)] for i in range(10)],
        "runtime": [0.1 * (i + 1) for i in range(10)],
        "memory": [100 * (i + 1) for i in range(10)],
        "code_size": [50 + i for i in range(10)],
    }
    pd.DataFrame(submission_stats_data).to_csv(
        DUMMY_SUBMISSION_STATS_CSV_TPL.format(lang=lang_for_demo.capitalize()),
        index=False,
    )
    print("Dummy data created.")


# --- Custom Collate Function ---
def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collates a list of dictionaries (from SubmissionDataset.__getitem__) into a batch dictionary.
    Strings are grouped into lists of strings. Tensors are stacked.
    """
    collated_batch: Dict[str, Any] = {}
    # Get keys from the first item, assuming all items have the same keys
    if not batch:
        return collated_batch

    first_item_keys = batch[0].keys()

    for key in first_item_keys:
        # Batch items based on their type
        if isinstance(batch[0][key], torch.Tensor):
            collated_batch[key] = torch.stack([item[key] for item in batch])
        elif isinstance(
            batch[0][key], str
        ):  # For 'code_str', 'statement_str', 'lang_str'
            collated_batch[key] = [item[key] for item in batch]
        elif isinstance(
            batch[0][key], (int, float)
        ):  # For other potential scalar values
            collated_batch[key] = torch.tensor([item[key] for item in batch])
        else:  # For any other types, just list them (e.g. verdict_raw_str)
            collated_batch[key] = [item[key] for item in batch]

    # 'lang_str' will be a list of identical strings, SubmissionPredictor expects a single lang string.
    # We'll assume all items in a batch have the same language.
    if (
        "lang_str" in collated_batch
        and isinstance(collated_batch["lang_str"], list)
        and collated_batch["lang_str"]
    ):
        collated_batch["lang_str_unified"] = collated_batch["lang_str"][0]

    return collated_batch


# --- Main Training Script ---
def main():
    print(f"Using device: {DEVICE}")
    if USE_DUMMY_DATA:
        create_dummy_data(TARGET_LANG)
        lang_suffix = TARGET_LANG.capitalize()
        problem_csv = DUMMY_PROBLEM_CSV
        submissions_code_csv = DUMMY_SUBMISSIONS_CODE_CSV_TPL.format(lang=lang_suffix)
        submission_stats_csv = DUMMY_SUBMISSION_STATS_CSV_TPL.format(lang=lang_suffix)
    else:  # Actual data paths
        lang_suffix = TARGET_LANG.capitalize()  # Adjust if your actual files differ
        problem_csv = ACTUAL_PROBLEM_CSV
        submissions_code_csv = ACTUAL_SUBMISSIONS_CODE_CSV_TPL.format(lang=lang_suffix)
        submission_stats_csv = ACTUAL_SUBMISSION_STATS_CSV_TPL.format(lang=lang_suffix)

    print(f"\n--- Loading data for language: {TARGET_LANG} ---")  # ... (path printing)

    print("\n--- Initializing components ---")
    preprocessor = Preprocessor()
    code_normalizer = CodeNormalizer()
    text_embedder = TextEmbedder(max_features=TEXT_EMB_MAX_FEATURES)
    print("Fitting TextEmbedder...")
    # Fit TextEmbedder with preprocessed text from the problem statements CSV
    df_probs_for_fitting = pd.read_csv(problem_csv)
    corpus_texts = (
        df_probs_for_fitting["statement"].fillna("").astype(str).str.strip()
        + " "
        + df_probs_for_fitting["input_spec"].fillna("").astype(str).str.strip()
        + " "
        + df_probs_for_fitting["output_spec"].fillna("").astype(str).str.strip()
    ).tolist()
    processed_corpus_for_fitting = [
        preprocessor.preprocess_text(text) for text in corpus_texts
    ]
    text_embedder.fit(processed_corpus_for_fitting)
    print(
        f"TextEmbedder fitted. Vocabulary size: {len(text_embedder.get_feature_names())}"
    )

    code_embedder_gnn = CodeEmbedderGNN(
        node_vocab_size=CODE_GNN_NODE_VOCAB_SIZE,
        node_embedding_dim=CODE_GNN_NODE_EMB_DIM,
        hidden_gnn_dim=CODE_GNN_HIDDEN_DIM,
        out_graph_embedding_dim=CODE_GNN_OUT_DIM,
        num_gnn_layers=CODE_GNN_LAYERS,
    ).to(
        DEVICE
    )  # Move GNN to device early

    concat_embedder = ConcatEmbedder(
        code_embedder=code_embedder_gnn,
        text_embedder=text_embedder,
        use_projection=CONCAT_USE_PROJECTION,
        projection_dim_scale_factor=CONCAT_PROJECTION_SCALE,
    ).to(
        DEVICE
    )  # Move ConcatEmbedder to device

    model = SubmissionPredictor(
        concat_embedder=concat_embedder,
        num_verdict_classes=NUM_VERDICT_CLASSES,
        mlp_hidden_dims=PREDICTOR_MLP_HIDDEN_DIMS,
    ).to(DEVICE)
    print(f"\nSubmissionPredictor model initialized on {DEVICE}.")

    print("\n--- Setting up Dataset and DataLoader ---")
    try:
        submission_dataset = SubmissionDataset(
            stats_csv_path=submission_stats_csv,
            code_csv_path=submissions_code_csv,
            problem_csv_path=problem_csv,
            preprocessor_instance=preprocessor,
            code_normalizer_instance=code_normalizer,
        )
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: CSV file not found during SubmissionDataset init: {e}")
        return
    except ValueError as e:
        print(f"CRITICAL ERROR: Could not initialize SubmissionDataset: {e}")
        return

    if len(submission_dataset) == 0:
        print("CRITICAL ERROR: Dataset is empty. Training cannot proceed.")
        return

    train_dataloader = DataLoader(
        submission_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn,  # Use the custom collate function
    )
    print(
        f"DataLoader created with {len(train_dataloader)} batches of size {BATCH_SIZE}."
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n--- Starting Training for {NUM_EPOCHS} epochs ---")
    model.train()

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        batches_processed = 0
        for batch_idx, batch_data in enumerate(train_dataloader):
            # batch_data is now a dictionary from custom_collate_fn
            codes_batch = batch_data["code_str"]  # List of code strings
            stmts_batch = batch_data["statement_str"]  # List of statement strings
            lang_batch_unified = batch_data[
                "lang_str_unified"
            ]  # Single lang string for the batch

            targets_verdict = batch_data["verdict_encoded"].to(DEVICE)

            optimizer.zero_grad()

            # Forward pass using SubmissionPredictor
            # Its internal ConcatEmbedder will handle code and text embedding
            verdict_logits = model(codes_batch, stmts_batch, lang_batch_unified)

            loss = criterion(verdict_logits, targets_verdict)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches_processed += 1
            if (batch_idx + 1) % 2 == 0 or (batch_idx + 1) == len(train_dataloader):
                print(
                    f"  Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}"
                )

        avg_epoch_loss = epoch_loss / batches_processed if batches_processed > 0 else 0
        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] completed. Average Loss: {avg_epoch_loss:.4f}"
        )

    print("\n--- Training Finished ---")

    print(f"\n--- Saving model state to {MODEL_SAVE_PATH} ---")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model state_dict saved.")

    # Saving TextEmbedder's vectorizer
    try:
        joblib.dump(text_embedder.vectorizer, TEXT_EMBEDDER_SAVE_PATH)
        print(f"TextEmbedder (vectorizer) saved to {TEXT_EMBEDDER_SAVE_PATH}")
    except Exception as e:
        print(f"Error saving TextEmbedder: {e}")

    # Saving CodeEmbedderGNN's vocabulary
    try:
        code_gnn_vocab_to_save = {
            "node_type_to_id": code_embedder_gnn.node_type_to_id,
            "next_node_type_id": code_embedder_gnn.next_node_type_id,
            "node_vocab_size": code_embedder_gnn.node_vocab_size,  # Original max size
        }
        with open(CODE_GNN_VOCAB_SAVE_PATH, "w") as f:
            json.dump(code_gnn_vocab_to_save, f, indent=4)
        print(f"CodeEmbedderGNN vocabulary saved to {CODE_GNN_VOCAB_SAVE_PATH}")
    except Exception as e:
        print(f"Error saving CodeEmbedderGNN vocabulary: {e}")

    if USE_DUMMY_DATA and os.path.exists(DUMMY_DATA_DIR):
        try:
            shutil.rmtree(DUMMY_DATA_DIR)
            print(f"\nCleaned up dummy data directory: {DUMMY_DATA_DIR}")
        except OSError as e:
            print(f"Error removing dummy data directory {DUMMY_DATA_DIR}: {e.strerror}")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e_main_imp:
        print(
            f"Exiting due to critical import errors before main execution: {e_main_imp}"
        )
    except Exception as e_main:
        print(
            f"An unexpected error occurred in main: {type(e_main).__name__} - {e_main}"
        )
        import traceback

        traceback.print_exc()
