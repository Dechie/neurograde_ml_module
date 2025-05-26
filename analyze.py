import os
import json
import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
import random

# --- Import Custom Modules ---
try:
    from preprocessor import Preprocessor
    from code_normalizer import CodeNormalizer
    from text_embedder import TextEmbedder
    from code_embedder import CodeEmbedderGNN
    from concat_embedder import ConcatEmbedder
    from submission_dataset import SubmissionDataset
    from submission_predictor import SubmissionPredictor
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}. Ensure all component .py files are accessible.")
    exit()

# --- Configuration ---
LANGUAGES_TO_ANALYZE = ["cpp", "python", "java"]  # Lowercase

SAVE_COMPONENTS_BASE_DIR = "saved_model_components"
GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE = os.path.join(
    SAVE_COMPONENTS_BASE_DIR, "global_text_embedder.joblib"
)
MODEL_FILENAME_TPL = "submission_predictor_{lang_key}.pth"
CODE_GNN_VOCAB_FILENAME_TPL = "code_gnn_vocab_{lang_key}.json"

PROBLEM_CSV_PATH = "data/final_problem_statements.csv"
SUBMISSIONS_CODE_CSV_TPL = "data/submissions_{lang_cap}.csv"
SUBMISSION_STATS_CSV_TPL = "data/submission_stats_{lang_cap}.csv"

# Model Hyperparameters (MUST match saved models)
CODE_GNN_NODE_VOCAB_SIZE_CONF = 2000
CODE_GNN_NODE_EMB_DIM_CONF = 64
CODE_GNN_HIDDEN_DIM_CONF = 128
CODE_GNN_OUT_DIM_CONF = 64
CODE_GNN_LAYERS_CONF = 2
CONCAT_USE_PROJECTION_CONF = True
CONCAT_PROJECTION_SCALE_CONF = 0.5
NUM_VERDICT_CLASSES_CONF = 7
PREDICTOR_MLP_HIDDEN_DIMS_CONF = [128, 64]

DEVICE_CONF = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Analysis Parameters
MAX_PROBLEMS_TO_COLLECT = 20
MAX_CORRECT_SAMPLES_PER_VERDICT_PER_LANG_PER_PROBLEM = (
    6  # Aim for this many for each verdict type
)

OUTPUT_CSV = "analysis_cross_language_diverse_correct_predictions.csv"

ID_TO_VERDICT_MAP_CONF = {
    0: "Accepted",
    1: "Wrong Answer",
    2: "Time Limit Exceeded",
    3: "Memory Limit Exceeded",
    4: "Runtime Error",
    5: "Compile Error",
    6: "Presentation Error",
    -1: "Unknown Verdict ID",
}
VERDICT_NAMES_ORDERED = [
    ID_TO_VERDICT_MAP_CONF[i] for i in range(NUM_VERDICT_CLASSES_CONF)
]


def load_model_and_components_for_lang(
    target_lang: str,
    global_text_embedder: TextEmbedder,
    preprocessor: Preprocessor,
    code_normalizer: CodeNormalizer,
) -> Tuple[Optional[SubmissionPredictor], Optional[SubmissionDataset]]:
    """Loads model, dataset, and other components for a single language."""
    print(f"  Loading components for language: {target_lang}...")
    lang_cap_for_file = target_lang.capitalize()
    lang_save_dir = os.path.join(SAVE_COMPONENTS_BASE_DIR, target_lang)

    model_file = os.path.join(
        lang_save_dir, MODEL_FILENAME_TPL.format(lang_key=target_lang)
    )
    code_gnn_vocab_file = os.path.join(
        lang_save_dir, CODE_GNN_VOCAB_FILENAME_TPL.format(lang_key=target_lang)
    )

    model_lang: Optional[SubmissionPredictor] = None
    dataset_lang: Optional[SubmissionDataset] = None

    try:
        if not os.path.exists(code_gnn_vocab_file):
            print(f"    CodeGNN vocab for {target_lang} not found. Skipping.")
            return None, None
        with open(code_gnn_vocab_file, "r") as f:
            code_gnn_vocab_data = json.load(f)
        code_embedder_gnn_lang = CodeEmbedderGNN(
            code_gnn_vocab_data.get("node_vocab_size", CODE_GNN_NODE_VOCAB_SIZE_CONF),
            CODE_GNN_NODE_EMB_DIM_CONF,
            CODE_GNN_HIDDEN_DIM_CONF,
            CODE_GNN_OUT_DIM_CONF,
            CODE_GNN_LAYERS_CONF,
        ).to(DEVICE_CONF)
        code_embedder_gnn_lang.node_type_to_id = code_gnn_vocab_data["node_type_to_id"]
        code_embedder_gnn_lang.next_node_type_id = code_gnn_vocab_data[
            "next_node_type_id"
        ]

        concat_embedder_lang = ConcatEmbedder(
            code_embedder_gnn_lang,
            global_text_embedder,
            CONCAT_USE_PROJECTION_CONF,
            CONCAT_PROJECTION_SCALE_CONF,
        ).to(DEVICE_CONF)

        if not os.path.exists(model_file):
            print(f"    Model file for {target_lang} not found. Skipping.")
            return None, None
        model_lang = SubmissionPredictor(
            concat_embedder_lang,
            NUM_VERDICT_CLASSES_CONF,
            PREDICTOR_MLP_HIDDEN_DIMS_CONF,
        ).to(DEVICE_CONF)
        model_lang.load_state_dict(torch.load(model_file, map_location=DEVICE_CONF))
        model_lang.eval()
        print(f"    Model for {target_lang} loaded.")

        submissions_code_csv = SUBMISSIONS_CODE_CSV_TPL.format(
            lang_cap=lang_cap_for_file
        )
        submission_stats_csv = SUBMISSION_STATS_CSV_TPL.format(
            lang_cap=lang_cap_for_file
        )
        if not (
            os.path.exists(submissions_code_csv)
            and os.path.exists(submission_stats_csv)
        ):
            print(f"    Dataset CSVs for {target_lang} not found. Skipping.")
            return model_lang, None  # Return model if it loaded

        dataset_lang = SubmissionDataset(
            submission_stats_csv,
            submissions_code_csv,
            PROBLEM_CSV_PATH,
            preprocessor,
            code_normalizer,
        )
        if len(dataset_lang) == 0:
            print(f"    Dataset for {target_lang} is empty. No samples.")
            return model_lang, None
        print(f"    Dataset for {target_lang} loaded with {len(dataset_lang)} samples.")

    except Exception as e:
        print(
            f"    ERROR loading components or dataset for {target_lang}: {e}. Skipping."
        )
        return None, None
    return model_lang, dataset_lang


# --- Main Analysis Script ---
def find_diverse_correct_predictions():
    print(
        f"--- Starting Analysis: Finding Problems with Diverse Correct Predictions ---"
    )
    print(f"Target languages: {', '.join(LANGUAGES_TO_ANALYZE)}")
    print(f"Device: {DEVICE_CONF}")

    # 1. Load shared components
    try:
        preprocessor = Preprocessor()
        code_normalizer = CodeNormalizer()
        print("Preprocessor and CodeNormalizer initialized.")

        if not os.path.exists(GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE):
            print(
                f"ERROR: Global TextEmbedder file NOT FOUND at '{GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE}'. Aborting."
            )
            return
        tfidf_vectorizer_fitted = joblib.load(GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE)
        global_text_embedder = TextEmbedder(vectorizer_model=tfidf_vectorizer_fitted)
        global_text_embedder._fitted = True
        if not (
            hasattr(global_text_embedder.vectorizer, "vocabulary_")
            and global_text_embedder.vectorizer.vocabulary_
        ):
            print("WARNING: Global TfidfVectorizer appears unfitted.")
            global_text_embedder._fitted = False
        if not global_text_embedder._fitted:
            print("ERROR: Global TextEmbedder not properly fitted. Aborting.")
            return
        print(
            f"Global TextEmbedder loaded. Vocab: {len(global_text_embedder.get_feature_names())}"
        )
    except Exception as e:
        print(f"Error initializing shared components: {e}")
        return

    # 2. Load models and datasets for all target languages
    models_by_lang: Dict[str, SubmissionPredictor] = {}
    datasets_by_lang: Dict[str, SubmissionDataset] = {}
    for lang in LANGUAGES_TO_ANALYZE:
        model, dataset = load_model_and_components_for_lang(
            lang, global_text_embedder, preprocessor, code_normalizer
        )
        if model and dataset:
            models_by_lang[lang] = model
            datasets_by_lang[lang] = dataset
        else:
            print(
                f"  Could not load full components for {lang}. It will be excluded from cross-language analysis."
            )

    if not models_by_lang or len(models_by_lang) < len(
        LANGUAGES_TO_ANALYZE
    ):  # Ensure all specified langs loaded
        print(
            "Not all languages had their models/datasets loaded successfully. Aborting detailed analysis."
        )
        # If you want to proceed with available languages, adjust this check.
        # For now, let's assume we need all specified languages.
        if (
            len(models_by_lang) < 2
        ):  # Need at least 2 to make cross-language meaningful for this script
            print(
                "Need at least two languages loaded to find cross-language problem examples. Aborting."
            )
            return

    # 3. Get all unique problem IDs present in *all* loaded datasets
    # This ensures we only consider problems that have data for every language we're analyzing
    problem_ids_per_lang: Dict[str, Set[str]] = {}
    for lang, ds in datasets_by_lang.items():
        problem_ids_per_lang[lang] = set(ds.df["problem_id"].astype(str).unique())

    if not problem_ids_per_lang:
        print("No problem IDs found in any dataset.")
        return

    common_problem_ids = set.intersection(*problem_ids_per_lang.values())
    if not common_problem_ids:
        print(
            "No common problem IDs found across all loaded language datasets. Cannot perform cross-language analysis."
        )
        return
    print(
        f"Found {len(common_problem_ids)} common problem IDs across loaded languages: {list(common_problem_ids)[:5]}..."
    )

    # 4. Iterate through common problems and collect desired samples
    all_collected_samples_for_csv: List[Dict] = []
    problems_fully_collected_count = 0

    # Iterate through problems in a shuffled order to get variety if we hit MAX_PROBLEMS_TO_COLLECT early
    shuffled_common_problem_ids = list(common_problem_ids)
    random.shuffle(shuffled_common_problem_ids)

    for problem_id in shuffled_common_problem_ids:
        if problems_fully_collected_count >= MAX_PROBLEMS_TO_COLLECT:
            print(
                f"Collected diverse samples for {MAX_PROBLEMS_TO_COLLECT} problems. Stopping."
            )
            break

        # For this problem, store correctly predicted samples per lang and per verdict
        # {lang: {verdict_str: [sample_info, ...]}}
        correct_preds_for_this_problem_by_lang_verdict = defaultdict(
            lambda: defaultdict(list)
        )

        # Check if this problem can satisfy the diversity criteria
        potential_to_satisfy_criteria = True  # Assume yes initially

        for lang in LANGUAGES_TO_ANALYZE:
            if lang not in models_by_lang or lang not in datasets_by_lang:
                continue  # Should not happen if check above passed

            model = models_by_lang[lang]
            dataset = datasets_by_lang[lang]

            # Filter dataset for current problem_id for this language
            problem_specific_indices = dataset.df[
                dataset.df["problem_id"].astype(str) == problem_id
            ].index
            if problem_specific_indices.empty:
                potential_to_satisfy_criteria = False
                break  # This lang has no data for this problem

            for i in problem_specific_indices:
                try:
                    item_data = dataset[i]
                except:
                    continue  # Skip if error getting item

                code_str = item_data["code_str"]
                statement_str = item_data["statement_str"]
                # lang_str_from_item = item_data['lang_str'] # Should be `lang`
                true_verdict_encoded = item_data["verdict_encoded"].item()
                true_verdict_raw = item_data["verdict_raw_str"]

                if true_verdict_encoded == -1:
                    continue

                with torch.no_grad():
                    verdict_logits = model([code_str], [statement_str], lang)
                probabilities = torch.softmax(verdict_logits, dim=1).squeeze()
                predicted_id = torch.argmax(probabilities).item()

                if predicted_id == true_verdict_encoded:
                    # Store enough info to reconstruct the sample later if chosen
                    if (
                        len(
                            correct_preds_for_this_problem_by_lang_verdict[lang][
                                true_verdict_raw
                            ]
                        )
                        < MAX_CORRECT_SAMPLES_PER_VERDICT_PER_LANG_PER_PROBLEM
                    ):
                        correct_preds_for_this_problem_by_lang_verdict[lang][
                            true_verdict_raw
                        ].append(
                            {
                                "problem_id": problem_id,
                                "submission_id": dataset.df.iloc[i].get(
                                    "submission_id", "N/A"
                                ),
                                "language_model_used": lang,
                                "code_original": dataset.df.iloc[i].get("code", "N/A"),
                                "true_verdict_string": true_verdict_raw,
                                "predicted_verdict_string": ID_TO_VERDICT_MAP_CONF.get(
                                    predicted_id, f"ID_{predicted_id}"
                                ),
                                # Add probabilities if needed: "probabilities": {n:p for n,p in zip(VERDICT_NAMES_ORDERED, probabilities.tolist())}
                            }
                        )

            # After checking all submissions for this lang for this problem,
            # if any verdict category for this language does not have enough samples,
            # this problem *might* not meet the strict criteria later.
            # The strict check: do we have 2 per verdict for *each* language for *this problem*?
            # Current modified goal: collect what we can per lang/verdict for this problem.

        # Now, check if this problem yielded enough diverse samples ACROSS languages
        # For the modified goal, we just collect everything found.
        # The original goal (6 per verdict type for this problem, 2 from each lang) is very strict.
        # Let's collect all correctly predicted samples we've gathered for this problem from the temp store.

        num_samples_for_this_problem = 0
        temp_problem_samples_to_add = []
        for (
            lang_key_collected,
            verdicts_map,
        ) in correct_preds_for_this_problem_by_lang_verdict.items():
            for verdict_str_collected, samples_list in verdicts_map.items():
                temp_problem_samples_to_add.extend(samples_list)
                num_samples_for_this_problem += len(samples_list)

        if (
            num_samples_for_this_problem > 0
        ):  # If we found any correct predictions for this problem
            # For this script's goal, let's check if we got a *decent variety* for this problem overall
            # e.g., at least N different verdict types represented among the correct predictions for this problem
            unique_verdicts_found_for_problem = set()
            for (
                lang_key_collected,
                verdicts_map,
            ) in correct_preds_for_this_problem_by_lang_verdict.items():
                for verdict_str_collected in verdicts_map.keys():
                    unique_verdicts_found_for_problem.add(verdict_str_collected)

            # Example diversity check: at least 3 different verdict types correctly predicted for this problem (across all langs)
            if len(unique_verdicts_found_for_problem) >= 3:
                all_collected_samples_for_csv.extend(temp_problem_samples_to_add)
                problems_fully_collected_count += 1
                print(
                    f"  Collected {num_samples_for_this_problem} diverse correct predictions for problem_id: {problem_id} ({len(unique_verdicts_found_for_problem)} unique verdict types). Total problems: {problems_fully_collected_count}/{MAX_PROBLEMS_TO_COLLECT}"
                )
            else:
                print(
                    f"  Problem {problem_id} did not yield enough diverse correct verdicts (found {len(unique_verdicts_found_for_problem)} types). Skipping."
                )

    # 5. Save to CSV
    if all_collected_samples_for_csv:
        df_collected = pd.DataFrame(all_collected_samples_for_csv)
        # Define column order if desired
        cols = [
            "problem_id",
            "language_model_used",
            "submission_id",
            "true_verdict_string",
            "predicted_verdict_string",
            "code_original",
        ]
        df_collected = df_collected[
            [c for c in cols if c in df_collected.columns]
            + [c for c in df_collected.columns if c not in cols]
        ]
        df_collected.to_csv(OUTPUT_CSV, index=False)
        print(
            f"\nSaved {len(df_collected)} selected samples from {problems_fully_collected_count} problems to: {OUTPUT_CSV}"
        )
    else:
        print(
            "\nNo samples met the diverse correct prediction criteria across problems."
        )

    print(f"\n--- Analysis Script Finished ---")


if __name__ == "__main__":
    try:
        if not LANGUAGES_TO_ANALYZE:
            raise ValueError("LANGUAGES_TO_ANALYZE is empty.")
        if not os.path.exists(PROBLEM_CSV_PATH):
            print(f"ERROR: Problem statements file missing: {PROBLEM_CSV_PATH}")
            exit()
        if not os.path.exists(GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE):
            print(
                f"ERROR: Global Text Embedder file missing: {GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE}"
            )
            exit()
        # Check existence of at least one language's model components to ensure setup is plausible
        if LANGUAGES_TO_ANALYZE:
            first_lang_dir = os.path.join(
                SAVE_COMPONENTS_BASE_DIR, LANGUAGES_TO_ANALYZE[0]
            )
            if not os.path.isdir(first_lang_dir):
                print(
                    f"ERROR: Saved components directory for first language '{LANGUAGES_TO_ANALYZE[0]}' not found at '{first_lang_dir}'. Ensure models are trained and paths are correct."
                )
                exit()
        find_diverse_correct_predictions()
    except Exception as e:
        print(f"An UNHANDLED error occurred: {type(e).__name__} - {e}")
        import traceback

        traceback.print_exc()
