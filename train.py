# train_model.py (Interleaved Multi-Language Training)

import os
import pandas as pd
import json
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from typing import Dict, List, Any, Optional  # Added Optional
import logging

# --- AMP Components ---
try:
    from torch.amp import GradScaler, autocast

    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    print("Warning: torch.amp not available. AMP will be disabled.")

    class GradScaler:  # type: ignore
        def __init__(self, device_type=None, enabled=False):
            self.enabled = enabled

        def scale(self, loss):
            return loss

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

    class autocast:  # type: ignore
        def __init__(self, device_type=None, enabled=False):
            self.enabled = enabled

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


# --- Custom Modules ---
try:
    from preprocessor import Preprocessor
    from code_normalizer import CodeNormalizer
    from text_embedder import TextEmbedder
    from code_embedder import CodeEmbedderGNN  # Ensure this filename is correct
    from concat_embedder import ConcatEmbedder
    from submission_dataset import SubmissionDataset
    from submission_predictor import SubmissionPredictor
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR in train_model.py: {e}")
    exit()

# --- Configuration Section ---
ACTUAL_PROBLEM_CSV = "data/final_problem_statements.csv"
ACTUAL_SUBMISSIONS_CODE_CSV_TPL = "data/submissions_{lang_cap}.csv"
ACTUAL_SUBMISSION_STATS_CSV_TPL = "data/submission_stats_{lang_cap}.csv"

LANGUAGES_TO_TRAIN = [
    "cpp",
    "python",
    # "java",
]  # <<< Languages to interleave training for

LOG_DIR = "training_logs"
# Single log file for all interleaved training, messages will be prefixed.
MASTER_LOG_FILE = os.path.join(LOG_DIR, "master_training_log.txt")

# Model & Training Hyperparameters (assumed mostly shared)
TEXT_EMB_MAX_FEATURES = 4000
CODE_GNN_NODE_VOCAB_SIZE = 2000
CODE_GNN_NODE_EMB_DIM = 128
CODE_GNN_HIDDEN_DIM = 256
CODE_GNN_OUT_DIM = 128
CODE_GNN_LAYERS = 3
CONCAT_USE_PROJECTION = True
CONCAT_PROJECTION_SCALE = 0.4
NUM_VERDICT_CLASSES = 7
PREDICTOR_MLP_HIDDEN_DIMS = [256, 256, 128, 64]
LEARNING_RATE = 5e-5
BATCH_SIZE = 32  # Batch size per language model
NUM_EPOCHS_PER_LANGUAGE_TURN = (
    1  # Number of epochs to run for a language before switching
)
TOTAL_INTERLEAVED_EPOCHS = 20  # Total "outer" epochs (each outer epoch trains all langs for NUM_EPOCHS_PER_LANGUAGE_TURN)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALIDATION_SPLIT_RATIO = 0.2
SAVE_COMPONENTS_BASE_DIR = (
    "saved_model_components"  # Base dir for lang-specific subdirs
)

NUM_EPOCHS = 30  # Total number of effective epochs PER LANGUAGE
DEFAULT_START_EPOCH = 0

NUM_WORKERS_DATALOADER = 2  # Adjusted based on typical Colab CPU, try 4 if beneficial
PIN_MEMORY_DATALOADER = True if DEVICE.type == "cuda" else False
PERSISTENT_WORKERS_DATALOADER = True if NUM_WORKERS_DATALOADER > 0 else False
USE_AMP = True if DEVICE.type == "cuda" and AMP_AVAILABLE else False

logger = None  # Global logger


# --- Logger Setup Function ---
def setup_master_logger(log_file_path: str, logger_name="MasterTrainingLogger"):
    master_logger = logging.getLogger(logger_name)
    master_logger.propagate = False
    master_logger.setLevel(logging.INFO)
    if not master_logger.handlers:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, mode="a")  # Append mode
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        master_logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )  # Generic console format
        console_handler.setFormatter(console_formatter)
        master_logger.addHandler(console_handler)
    return master_logger


# --- Custom Collate Function (as before) ---
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


# --- Main Training Script ---
def main():
    global logger
    logger = setup_master_logger(MASTER_LOG_FILE)

    logger.info(
        f"========== Starting Interleaved Multi-Language Training Process =========="
    )
    logger.info(f"Languages to train: {', '.join(LANGUAGES_TO_TRAIN)}")
    logger.info(f"Selected device: {DEVICE}")
    if DEVICE.type == "cuda":
        logger.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    if USE_AMP:
        logger.info("Automatic Mixed Precision (AMP) ENABLED.")
    else:
        logger.info("Automatic Mixed Precision (AMP) DISABLED.")

    # --- Initialize Preprocessor and CodeNormalizer (Shared) ---
    preprocessor = Preprocessor()
    code_normalizer = CodeNormalizer()
    logger.info("Shared Preprocessor and CodeNormalizer initialized.")

    # --- Fit a SINGLE TextEmbedder on the full problem corpus ---
    # This TextEmbedder will be shared by all language models if problem statements are the same.
    # If each language has vastly different problem statement styles/vocab,
    # you might consider language-specific TextEmbedders, but that adds complexity.
    global_text_embedder = TextEmbedder(max_features=TEXT_EMB_MAX_FEATURES)
    global_text_embedder_fitted_path = os.path.join(
        SAVE_COMPONENTS_BASE_DIR, "global_text_embedder.joblib"
    )  # Save it globally
    global_text_embedder_loaded = False

    if os.path.exists(global_text_embedder_fitted_path):
        try:
            loaded_vectorizer = joblib.load(global_text_embedder_fitted_path)
            global_text_embedder.vectorizer = loaded_vectorizer
            global_text_embedder._fitted = True
            global_text_embedder_loaded = True
            logger.info(
                f"Global TextEmbedder LOADED from {global_text_embedder_fitted_path}. Vocab: {len(global_text_embedder.get_feature_names())}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to load Global TextEmbedder from {global_text_embedder_fitted_path}. Fitting from scratch. Error: {e}"
            )

    if not global_text_embedder_loaded:
        logger.info(
            "Fitting Global TextEmbedder from scratch on all problem statements..."
        )
        try:
            df_probs_for_fitting = pd.read_csv(ACTUAL_PROBLEM_CSV)
        except FileNotFoundError:
            logger.error(
                f"CRITICAL: Problem statements file '{ACTUAL_PROBLEM_CSV}' not found for Global TextEmbedder fitting."
            )
            return
        corpus_texts = (
            df_probs_for_fitting["statement"].fillna("")
            + " "
            + df_probs_for_fitting["input_spec"].fillna("")
            + " "
            + df_probs_for_fitting["output_spec"].fillna("")
        ).tolist()
        processed_corpus_for_fitting = [
            preprocessor.preprocess_text(text) for text in corpus_texts
        ]
        if not processed_corpus_for_fitting:
            logger.error("CRITICAL: No text to fit Global TextEmbedder.")
            return
        global_text_embedder.fit(processed_corpus_for_fitting)
        logger.info(
            f"Global TextEmbedder fitted. Vocab: {len(global_text_embedder.get_feature_names())}"
        )
        try:  # Save the globally fitted text_embedder
            joblib.dump(
                global_text_embedder.vectorizer, global_text_embedder_fitted_path
            )
            logger.info(
                f"Global TextEmbedder (vectorizer) saved to {global_text_embedder_fitted_path}"
            )
        except Exception as e:
            logger.error(f"Error saving globally fitted TextEmbedder: {e}")

    # --- Dictionaries to hold language-specific components ---
    models: Dict[str, SubmissionPredictor] = {}
    optimizers: Dict[str, optim.Optimizer] = {}
    schedulers: Dict[str, optim.lr_scheduler._LRScheduler] = {}
    train_dataloaders: Dict[str, DataLoader] = {}
    val_dataloaders: Dict[str, Optional[DataLoader]] = {}
    code_embedders_gnn: Dict[str, CodeEmbedderGNN] = {}  # To save vocab
    # TextEmbedder is global, ConcatEmbedder is per model

    # Language-specific states for resuming
    start_epochs: Dict[str, int] = {
        lang: DEFAULT_START_EPOCH for lang in LANGUAGES_TO_TRAIN
    }
    best_val_losses: Dict[str, float] = {
        lang: float("inf") for lang in LANGUAGES_TO_TRAIN
    }

    # --- Initialize components for each language ---
    for lang_key in LANGUAGES_TO_TRAIN:
        logger.info(
            f"\n--- Initializing components for language: {lang_key.upper()} ---"
        )
        safe_target_lang = lang_key  # Already lowercase from LANGUAGES_TO_TRAIN
        lang_cap_for_file = safe_target_lang.capitalize()
        lang_save_dir = os.path.join(SAVE_COMPONENTS_BASE_DIR, safe_target_lang)
        os.makedirs(lang_save_dir, exist_ok=True)

        model_save_path = os.path.join(
            lang_save_dir, f"submission_predictor_{safe_target_lang}.pth"
        )
        # text_embedder_save_path is now global_text_embedder_fitted_path
        code_gnn_vocab_save_path = os.path.join(
            lang_save_dir, f"code_gnn_vocab_{safe_target_lang}.json"
        )
        checkpoint_file = os.path.join(
            lang_save_dir, f"training_checkpoint_{safe_target_lang}.pth"
        )

        # CodeEmbedderGNN (language-specific vocab and weights)
        code_gnn = CodeEmbedderGNN(
            CODE_GNN_NODE_VOCAB_SIZE,
            CODE_GNN_NODE_EMB_DIM,
            CODE_GNN_HIDDEN_DIM,
            CODE_GNN_OUT_DIM,
            CODE_GNN_LAYERS,
        ).to(DEVICE)
        if os.path.exists(code_gnn_vocab_save_path):
            try:
                with open(code_gnn_vocab_save_path, "r") as f:
                    code_gnn_vocab_data = json.load(f)
                code_gnn.node_type_to_id = code_gnn_vocab_data["node_type_to_id"]
                code_gnn.next_node_type_id = code_gnn_vocab_data["next_node_type_id"]
                logger.info(
                    f"CodeGNN vocab for {lang_key} LOADED. Items: {len(code_gnn.node_type_to_id)}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load CodeGNN vocab for {lang_key}. Using fresh. Error: {e}"
                )
        code_embedders_gnn[lang_key] = code_gnn  # Store for saving vocab later

        # ConcatEmbedder (uses global TextEmbedder, language-specific CodeEmbedderGNN)
        concat_emb = ConcatEmbedder(
            code_gnn,
            global_text_embedder,
            CONCAT_USE_PROJECTION,
            CONCAT_PROJECTION_SCALE,
        ).to(DEVICE)

        # SubmissionPredictor
        model = SubmissionPredictor(
            concat_emb, NUM_VERDICT_CLASSES, PREDICTOR_MLP_HIDDEN_DIMS
        ).to(DEVICE)
        models[lang_key] = model
        logger.info(
            f"Model for {lang_key} initialized. Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

        optimizers[lang_key] = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        schedulers[lang_key] = optim.lr_scheduler.ReduceLROnPlateau(
            optimizers[lang_key], mode="min", factor=0.1, patience=3
        )

        if os.path.exists(checkpoint_file):
            logger.info(
                f"Attempting to load checkpoint for {lang_key}: {checkpoint_file}"
            )
            try:
                checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
                models[lang_key].load_state_dict(checkpoint["model_state_dict"])
                optimizers[lang_key].load_state_dict(checkpoint["optimizer_state_dict"])
                start_epochs[lang_key] = checkpoint["epoch"] + 1
                best_val_losses[lang_key] = checkpoint.get(
                    "best_val_loss", float("inf")
                )
                if "scheduler_state_dict" in checkpoint:
                    schedulers[lang_key].load_state_dict(
                        checkpoint["scheduler_state_dict"]
                    )
                if (
                    "amp_scaler_state_dict" in checkpoint
                    and checkpoint["amp_scaler_state_dict"] is not None
                    and USE_AMP
                ):
                    # Scaler is global for the script, state loaded per language if different, or one global scaler
                    # For simplicity with interleaved, one global scaler might be fine if always enabled/disabled same way
                    # If loading per-lang scaler state: ensure scaler is re-init or state loaded correctly
                    # This example assumes a global scaler, so state might not be perfectly resumed for scaler if languages differ in AMP use
                    # For now, we'll load it if it exists, assuming global scaler can handle it
                    pass  # Scaler state loaded globally before outer epoch loop
                logger.info(
                    f"Checkpoint for {lang_key} LOADED. Resuming from epoch {start_epochs[lang_key]}. Best val loss: {best_val_losses[lang_key]:.4f}"
                )
            except Exception as e:
                logger.warning(
                    f"Could not load checkpoint for {lang_key}. Starting fresh. Error: {e}"
                )
                start_epochs[lang_key] = DEFAULT_START_EPOCH
                best_val_losses[lang_key] = float("inf")
        else:
            logger.info(
                f"No checkpoint for {lang_key}. Starting fresh from epoch {DEFAULT_START_EPOCH + 1}."
            )
            start_epochs[lang_key] = DEFAULT_START_EPOCH

        # Dataset and DataLoader for this language
        problem_csv_for_dataset = ACTUAL_PROBLEM_CSV  # Shared problem statements
        submissions_code_csv_for_dataset = ACTUAL_SUBMISSIONS_CODE_CSV_TPL.format(
            lang_cap=lang_cap_for_file
        )
        submission_stats_csv_for_dataset = ACTUAL_SUBMISSION_STATS_CSV_TPL.format(
            lang_cap=lang_cap_for_file
        )

        try:
            full_ds = SubmissionDataset(
                submission_stats_csv_for_dataset,
                submissions_code_csv_for_dataset,
                problem_csv_for_dataset,
                preprocessor,
                code_normalizer,
            )
        except Exception as e:
            logger.error(f"CRITICAL Dataset init error for {lang_key}: {e}")
            continue  # Skip this lang
        if len(full_ds) == 0:
            logger.error(f"CRITICAL: Full dataset for {lang_key} empty.")
            continue  # Skip

        num_total = len(full_ds)
        num_val = int(VALIDATION_SPLIT_RATIO * num_total)
        if num_total > 0 and num_val == 0 and VALIDATION_SPLIT_RATIO > 0:
            num_val = 1
        num_train = num_total - num_val

        if num_train <= 0:
            logger.error(
                f"CRITICAL: Not enough data for {lang_key} training (Train: {num_train})."
            )
            continue

        train_ds, val_ds = (
            random_split(full_ds, [num_train, num_val])
            if num_val > 0
            else (full_ds, None)
        )
        logger.info(
            f"Data for {lang_key}: Total: {num_total}, Train: {len(train_ds)}, Val: {len(val_ds) if val_ds else 0}"
        )

        train_dataloaders[lang_key] = DataLoader(
            train_ds,
            BATCH_SIZE,
            shuffle=True,
            collate_fn=custom_collate_fn,
            num_workers=NUM_WORKERS_DATALOADER,
            pin_memory=PIN_MEMORY_DATALOADER,
            persistent_workers=PERSISTENT_WORKERS_DATALOADER,
        )
        val_dataloaders[lang_key] = (
            DataLoader(
                val_ds,
                BATCH_SIZE,
                shuffle=False,
                collate_fn=custom_collate_fn,
                num_workers=NUM_WORKERS_DATALOADER,
                pin_memory=PIN_MEMORY_DATALOADER,
                persistent_workers=PERSISTENT_WORKERS_DATALOADER,
            )
            if val_ds
            else None
        )
        logger.info(
            f"DataLoaders for {lang_key}: Train batches: {len(train_dataloaders[lang_key])}, Val batches: {len(val_dataloaders[lang_key]) if val_ds else 'N/A'}"
        )

    # --- Initialize Global GradScaler for AMP ---
    # Scaler state can be loaded from the first language's checkpoint that has it, if resuming.
    # Or just initialize fresh if no relevant checkpoint has scaler state.
    global_scaler = GradScaler(DEVICE.type, enabled=USE_AMP)
    # Attempt to load scaler state from the first language's checkpoint if USE_AMP
    if USE_AMP and LANGUAGES_TO_TRAIN:
        first_lang_key = LANGUAGES_TO_TRAIN[0]
        first_lang_checkpoint_file = os.path.join(
            SAVE_COMPONENTS_BASE_DIR,
            first_lang_key,
            f"training_checkpoint_{first_lang_key}.pth",
        )
        if os.path.exists(first_lang_checkpoint_file):
            try:
                checkpoint = torch.load(first_lang_checkpoint_file, map_location=DEVICE)
                if (
                    "amp_scaler_state_dict" in checkpoint
                    and checkpoint["amp_scaler_state_dict"] is not None
                ):
                    global_scaler.load_state_dict(checkpoint["amp_scaler_state_dict"])
                    logger.info(
                        f"Global AMP GradScaler state LOADED from checkpoint of {first_lang_key}."
                    )
            except Exception as e:
                logger.warning(
                    f"Could not load global AMP scaler state from {first_lang_key}'s checkpoint. Using fresh scaler. Error: {e}"
                )

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    logger.info(
        f"\n--- Starting Interleaved Training for {TOTAL_INTERLEAVED_EPOCHS} total outer epochs ---"
    )
    logger.info(
        f"--- Each language will be trained for {NUM_EPOCHS_PER_LANGUAGE_TURN} epoch(s) per turn ---"
    )

    for outer_epoch in range(TOTAL_INTERLEAVED_EPOCHS):
        logger.info(
            f"***** Outer Epoch [{outer_epoch + 1}/{TOTAL_INTERLEAVED_EPOCHS}] *****"
        )
        for lang_key in LANGUAGES_TO_TRAIN:
            if lang_key not in models or lang_key not in train_dataloaders:
                logger.warning(
                    f"Skipping language {lang_key} for Outer Epoch {outer_epoch+1} due to missing model or dataloader."
                )
                continue

            model = models[lang_key]
            optimizer = optimizers[lang_key]
            scheduler = schedulers[lang_key]
            train_loader = train_dataloaders[lang_key]
            val_loader = val_dataloaders.get(lang_key)  # Might be None

            # Determine current epoch number for this language based on how many times it has been trained
            # This needs careful tracking if NUM_EPOCHS_PER_LANGUAGE_TURN > 1 or if resuming.
            # For simplicity, let's assume start_epochs[lang_key] is the next epoch to run for this language.
            current_lang_epoch_start = start_epochs[lang_key]
            current_lang_epoch_end = (
                current_lang_epoch_start + NUM_EPOCHS_PER_LANGUAGE_TURN
            )

            logger.info(
                f"--- Training language: {lang_key.upper()} for inner epoch(s) {current_lang_epoch_start + 1} to {current_lang_epoch_end} ---"
            )

            for inner_epoch_idx in range(NUM_EPOCHS_PER_LANGUAGE_TURN):
                actual_epoch_num_for_lang = current_lang_epoch_start + inner_epoch_idx
                if (
                    actual_epoch_num_for_lang >= NUM_EPOCHS
                ):  # Check against total desired epochs for this language
                    logger.info(
                        f"Language {lang_key.upper()} has completed its total {NUM_EPOCHS} epochs. Skipping."
                    )
                    break

                model.train()
                epoch_train_loss = 0.0
                train_batches_processed = 0
                for batch_idx, batch_data in enumerate(train_loader):
                    codes_batch = batch_data["code_str"]
                    stmts_batch = batch_data["statement_str"]
                    lang_for_batch = batch_data[
                        "lang_str_unified"
                    ]  # This should be lang_key
                    targets_verdict = batch_data["verdict_encoded"].to(DEVICE)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast(DEVICE.type, enabled=USE_AMP):
                        verdict_logits = model(codes_batch, stmts_batch, lang_for_batch)
                        loss = criterion(verdict_logits, targets_verdict)
                    if torch.isnan(loss):
                        logger.warning(
                            f"NaN loss for {lang_key} at lang-epoch {actual_epoch_num_for_lang+1}, batch {batch_idx+1}. Skipping."
                        )
                        continue
                    global_scaler.scale(loss).backward()
                    global_scaler.step(optimizer)
                    global_scaler.update()
                    epoch_train_loss += loss.item()
                    train_batches_processed += 1
                    if (batch_idx + 1) % (
                        max(1, (len(train_loader) + 9) // 10)
                    ) == 0 or (batch_idx + 1) == len(train_loader):
                        logger.info(
                            f"  Lang: {lang_key.upper()}, Epoch(Lang) [{actual_epoch_num_for_lang+1}/{NUM_EPOCHS}], Train Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                        )
                avg_epoch_train_loss = (
                    epoch_train_loss / train_batches_processed
                    if train_batches_processed > 0
                    else float("nan")
                )
                logger.info(
                    f"Lang: {lang_key.upper()}, Epoch(Lang) [{actual_epoch_num_for_lang+1}/{NUM_EPOCHS}] Training completed. Avg Train Loss: {avg_epoch_train_loss:.4f}"
                )

                current_val_loss_for_epoch = float("inf")
                if val_loader:
                    model.eval()
                    epoch_val_loss = 0.0
                    val_batches_processed_samples = 0
                    correct_val_predictions = 0
                    total_val_samples_for_acc = 0
                    with torch.no_grad():
                        for batch_data_val in val_loader:
                            # ... (validation batch processing as before)
                            codes_val = batch_data_val["code_str"]
                            stmts_val = batch_data_val["statement_str"]
                            lang_val = batch_data_val["lang_str_unified"]
                            targets_verdict_val = batch_data_val["verdict_encoded"].to(
                                DEVICE
                            )
                            valid_targets_mask = targets_verdict_val != -1
                            if not torch.any(valid_targets_mask):
                                continue
                            with autocast(DEVICE.type, enabled=USE_AMP):
                                verdict_logits_val = model(
                                    codes_val, stmts_val, lang_val
                                )
                                val_loss_item = criterion(
                                    verdict_logits_val[valid_targets_mask],
                                    targets_verdict_val[valid_targets_mask],
                                ).item()
                            epoch_val_loss += (
                                val_loss_item * valid_targets_mask.sum().item()
                            )
                            val_batches_processed_samples += (
                                valid_targets_mask.sum().item()
                            )
                            _, predicted_classes = torch.max(
                                verdict_logits_val[valid_targets_mask], 1
                            )
                            correct_val_predictions += (
                                (
                                    predicted_classes
                                    == targets_verdict_val[valid_targets_mask]
                                )
                                .sum()
                                .item()
                            )
                            total_val_samples_for_acc += valid_targets_mask.sum().item()
                    avg_epoch_val_loss = (
                        epoch_val_loss / val_batches_processed_samples
                        if val_batches_processed_samples > 0
                        else float("nan")
                    )
                    current_val_loss_for_epoch = avg_epoch_val_loss
                    val_accuracy = (
                        (correct_val_predictions / total_val_samples_for_acc * 100)
                        if total_val_samples_for_acc > 0
                        else 0.0
                    )
                    logger.info(
                        f"Lang: {lang_key.upper()}, Epoch(Lang) [{actual_epoch_num_for_lang+1}/{NUM_EPOCHS}] Val Loss: {avg_epoch_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
                    )
                    if not pd.isna(avg_epoch_val_loss):
                        schedulers[lang_key].step(avg_epoch_val_loss)

                    if (
                        not pd.isna(avg_epoch_val_loss)
                        and avg_epoch_val_loss < best_val_losses[lang_key]
                    ):
                        logger.info(
                            f"  Lang: {lang_key.upper()}, Val loss improved ({best_val_losses[lang_key]:.4f} --> {avg_epoch_val_loss:.4f}). Saving checkpoint..."
                        )
                        best_val_losses[lang_key] = avg_epoch_val_loss
                        # Paths specific to this language
                        lang_model_save_path = os.path.join(
                            SAVE_COMPONENTS_BASE_DIR,
                            lang_key,
                            f"submission_predictor_{lang_key}.pth",
                        )
                        lang_text_emb_save_path = (
                            global_text_embedder_fitted_path  # Global text embedder
                        )
                        lang_code_gnn_vocab_save_path = os.path.join(
                            SAVE_COMPONENTS_BASE_DIR,
                            lang_key,
                            f"code_gnn_vocab_{lang_key}.json",
                        )
                        lang_checkpoint_file = os.path.join(
                            SAVE_COMPONENTS_BASE_DIR,
                            lang_key,
                            f"training_checkpoint_{lang_key}.pth",
                        )

                        checkpoint_data = {
                            "epoch": actual_epoch_num_for_lang,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_val_loss": best_val_losses[lang_key],
                            "scheduler_state_dict": scheduler.state_dict(),
                            "amp_scaler_state_dict": (
                                global_scaler.state_dict() if USE_AMP else None
                            ),
                        }
                        torch.save(checkpoint_data, lang_checkpoint_file)
                        torch.save(model.state_dict(), lang_model_save_path)
                        # Global text embedder already saved once. No need to save per language unless it's lang-specific.
                        code_gnn_vocab_to_save = {
                            "node_type_to_id": code_embedders_gnn[
                                lang_key
                            ].node_type_to_id,
                            "next_node_type_id": code_embedders_gnn[
                                lang_key
                            ].next_node_type_id,
                            "node_vocab_size": code_embedders_gnn[
                                lang_key
                            ].node_vocab_size,
                        }
                        with open(lang_code_gnn_vocab_save_path, "w") as f:
                            json.dump(code_gnn_vocab_to_save, f, indent=4)
                        logger.info(
                            f"  Checkpoint and best components for {lang_key} saved for lang-epoch {actual_epoch_num_for_lang+1}."
                        )

                # Update the starting epoch for the next turn of this language
                start_epochs[lang_key] = actual_epoch_num_for_lang + 1

            # End of inner epoch loop for one language's turn

        # End of outer epoch loop (all languages trained for one turn)
        # Check if all languages have completed their total NUM_EPOCHS
        all_langs_completed = all(
            start_epochs[lang] >= NUM_EPOCHS
            for lang in LANGUAGES_TO_TRAIN
            if lang in start_epochs
        )
        if all_langs_completed:
            logger.info(
                f"All languages have completed {NUM_EPOCHS} epochs. Stopping interleaved training."
            )
            break

    logger.info(f"--- Interleaved Training Finished ---")
    for lang_key in LANGUAGES_TO_TRAIN:
        if lang_key in models:  # Only log if model was successfully initialized
            logger.info(
                f"Find logs in: {MASTER_LOG_FILE} (search for [{lang_key.upper()}])"
            )
            logger.info(
                f"Find saved components for {lang_key} in: {os.path.join(SAVE_COMPONENTS_BASE_DIR, lang_key)}/"
            )


# --- Main Execution ---
if __name__ == "__main__":
    try:
        if not LANGUAGES_TO_TRAIN:  # Ensure list is not empty
            raise ValueError(
                "LANGUAGES_TO_TRAIN configuration is empty at the top of the script."
            )
        main()
    except ImportError as e_main_imp:
        print(f"Exiting due to critical top-level import errors: {e_main_imp}")
    except ValueError as ve_main:
        print(f"Configuration error: {ve_main}")
    except Exception as e_main:
        if logger:
            logger.critical(
                f"An UNHANDLED error: {type(e_main).__name__} - {e_main}", exc_info=True
            )
        else:
            print(
                f"An UNHANDLED error (logger not available): {type(e_main).__name__} - {e_main}"
            )
            import traceback

            traceback.print_exc()
