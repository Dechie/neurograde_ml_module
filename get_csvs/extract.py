# extract.py (Strategy 2: Two-Pass Global Quotas)

import pandas as pd
import os
import random
from tqdm import tqdm
from collections import Counter  # For counting verdict occurrences
from typing import List, Dict
import numpy as np

# Assuming CodeNormalizer is in the same directory or path
try:
    from code_normalizer import CodeNormalizer
except ImportError:
    print(
        "CRITICAL: CodeNormalizer class not found. Ensure code_normalizer.py is accessible."
    )

    # Define a dummy if not found so script can be parsed
    class CodeNormalizer:
        @staticmethod
        def normalize_file(filepath, lang=None):
            return f"Normalized content of {filepath}"

        def normalize_code(self, code, lang=None):
            return code.strip()


# --- Configuration ---
PROBLEM_STATEMENTS_CSV = "data/final_problem_statements.csv"  # Ensure this exists
PROJECT_CODENET_BASE_DIR = "../Project_CodeNet/"  # Path to the root of CodeNet dataset
CODENET_METADATA_PER_PROBLEM_DIR = os.path.join(PROJECT_CODENET_BASE_DIR, "metadata")
CODENET_DATA_DIR = os.path.join(PROJECT_CODENET_BASE_DIR, "data")

OUTPUT_DIR = (
    "data/"  # Where your submissions_Lang.csv and submission_stats_Lang.csv will go
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_LANGUAGES_INFO = {
    "C++": {
        "code_csv": os.path.join(OUTPUT_DIR, "submissions_Cpp.csv"),
        "stats_csv": os.path.join(OUTPUT_DIR, "submission_stats_Cpp.csv"),
        "codenet_dir_name": "C++",
    },
    "Python": {
        "code_csv": os.path.join(OUTPUT_DIR, "submissions_Python.csv"),
        "stats_csv": os.path.join(OUTPUT_DIR, "submission_stats_Python.csv"),
        "codenet_dir_name": "Python",
    },
    "Java": {
        "code_csv": os.path.join(OUTPUT_DIR, "submissions_Java.csv"),
        "stats_csv": os.path.join(OUTPUT_DIR, "submission_stats_Java.csv"),
        "codenet_dir_name": "Java",
    },
}

GLOBAL_TARGET_VERDICTS = [
    "Accepted",
    "Compile Error",
    "Memory Limit Exceeded",
    "Presentation Error",
    "Runtime Error",
    "Time Limit Exceeded",
    "Wrong Answer",
]
RANDOM_SEED = 42
METADATA_REQUIRED_COLS = [
    "submission_id",
    "problem_id",
    "language",
    "filename_ext",
    "status",
    "cpu_time",
    "memory",
    "code_size",
]

# --- NEW SAMPLING PARAMETERS FOR STRATEGY 2 ---
GLOBAL_SAMPLING_TARGETS_PER_LANG = {
    # These are examples, TUNE THEM based on your data and desired balance!
    # 'cap_at' is an absolute maximum for a class.
    "Accepted": {"strategy": "max_samples", "value": 15000},  # Cap majority class
    "Wrong Answer": {
        "strategy": "max_samples",
        "value": 12000,
    },  # Cap another large class
    "Time Limit Exceeded": {
        "strategy": "min_samples_or_all",
        "value": 40000,
    },  # Try for 4k, or take all if less
    "Memory Limit Exceeded": {"strategy": "min_samples_or_all", "value": 30000},
    "Runtime Error": {"strategy": "min_samples_or_all", "value": 40000},
    "Compile Error": {"strategy": "min_samples_or_all", "value": 30000},
    "Presentation Error": {"strategy": "min_samples_or_all", "value": 20000},
}
# Fallback for verdicts not explicitly listed above
DEFAULT_SAMPLING_STRATEGY_PER_LANG = {
    "strategy": "min_samples_or_all",
    "value": 1000,
}  # Ensure even unlisted get some representation

# Maximum total submissions to aim for per language (optional, can act as a global cap)
# Set to None if you want the sum of GLOBAL_SAMPLING_TARGETS_PER_LANG to dictate the total
MAX_TOTAL_SAMPLES_PER_LANGUAGE = None  # e.g., 40000


# --- Helper Functions ---
def map_codenet_language_to_target(codenet_lang_str):
    if not isinstance(codenet_lang_str, str):
        return None
    lang_lower = codenet_lang_str.lower()
    if "c++" in lang_lower or "cpp" in lang_lower:
        return "C++"
    if "python" in lang_lower or "pypy" in lang_lower:
        return "Python"
    if "java" in lang_lower:
        return "Java"
    return None


def map_codenet_status_to_target_verdict(codenet_status_str):
    if not isinstance(codenet_status_str, str):
        return None
    if codenet_status_str == "Accepted":
        return "Accepted"
    if codenet_status_str == "Compile Error":
        return "Compile Error"
    if codenet_status_str == "Memory Limit Exceeded":
        return "Memory Limit Exceeded"
    if codenet_status_str == "WA: Presentation Error":
        return "Presentation Error"
    if codenet_status_str == "Time Limit Exceeded":
        return "Time Limit Exceeded"
    if codenet_status_str.startswith("Runtime Error"):
        return "Runtime Error"  # Handles variants
    if codenet_status_str == "Wrong Answer":
        return "Wrong Answer"
    return None


# --- Main Logic ---
def main():
    random.seed(RANDOM_SEED)

    try:
        problem_meta_df = pd.read_csv(PROBLEM_STATEMENTS_CSV)
        user_problem_ids = problem_meta_df["problem_id"].unique().tolist()
        if not user_problem_ids:
            print(f"No problem_ids in {PROBLEM_STATEMENTS_CSV}. Exiting.")
            return
        print(
            f"Loaded {len(user_problem_ids)} unique problem IDs from {PROBLEM_STATEMENTS_CSV}"
        )
    except Exception as e:
        print(f"Error loading {PROBLEM_STATEMENTS_CSV}: {e}")
        return

    all_problem_metadata_dfs = []
    print("Loading CodeNet metadata (this might take a while)...")
    for p_id in tqdm(
        user_problem_ids, desc="Loading metadata per problem"
    ):  # tqdm for metadata loading
        fpath = os.path.join(CODENET_METADATA_PER_PROBLEM_DIR, f"{p_id}.csv")
        try:
            all_problem_metadata_dfs.append(
                pd.read_csv(fpath, usecols=METADATA_REQUIRED_COLS, low_memory=False)
            )
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Warning: Error reading {fpath}: {e}")

    if not all_problem_metadata_dfs:
        print("No CodeNet metadata loaded. Exiting.")
        return
    codenet_full_meta_df = pd.concat(all_problem_metadata_dfs, ignore_index=True)
    print(
        f"Loaded metadata for {codenet_full_meta_df['problem_id'].nunique()} problems, {len(codenet_full_meta_df)} total entries."
    )

    print("Preprocessing CodeNet metadata...")
    codenet_full_meta_df["target_language"] = codenet_full_meta_df["language"].apply(
        map_codenet_language_to_target
    )
    codenet_full_meta_df["target_verdict"] = codenet_full_meta_df["status"].apply(
        map_codenet_status_to_target_verdict
    )
    codenet_processed_meta_df = codenet_full_meta_df.dropna(
        subset=["target_language", "target_verdict"]
    ).copy()
    codenet_processed_meta_df = codenet_processed_meta_df[
        codenet_processed_meta_df["target_verdict"].isin(GLOBAL_TARGET_VERDICTS)
    ]
    print(
        f"Filtered to {len(codenet_processed_meta_df)} relevant CodeNet metadata entries with target verdicts and languages."
    )
    if codenet_processed_meta_df.empty:
        print("No relevant CodeNet submissions found after initial filter. Exiting.")
        return

    code_normalizer_instance = CodeNormalizer()

    for lang_name_key, lang_details in TARGET_LANGUAGES_INFO.items():
        code_output_csv, stats_output_csv = (
            lang_details["code_csv"],
            lang_details["stats_csv"],
        )
        codenet_data_subdir = lang_details["codenet_dir_name"]

        print(f"\n--- Processing language: {lang_name_key} ---")

        lang_specific_meta_df = codenet_processed_meta_df[
            codenet_processed_meta_df["target_language"] == lang_name_key
        ].copy()
        if lang_specific_meta_df.empty:
            print(
                f"No submissions found for '{lang_name_key}' after filtering. Skipping."
            )
            continue

        print(
            f"Found {len(lang_specific_meta_df)} total submissions for {lang_name_key}."
        )

        # 1. Calculate current global availability for this language
        global_verdict_availability = Counter(lang_specific_meta_df["target_verdict"])
        print(f"Global verdict availability for {lang_name_key}:")
        for verdict, count in global_verdict_availability.items():
            print(f"  {verdict}: {count}")

        # 2. Determine target number of samples for each verdict
        target_samples_per_verdict_for_lang: Dict[str, int] = {}
        max_class_count = (
            max(global_verdict_availability.values())
            if global_verdict_availability
            else 0
        )

        for verdict in GLOBAL_TARGET_VERDICTS:
            config = GLOBAL_SAMPLING_TARGETS_PER_LANG.get(
                verdict, DEFAULT_SAMPLING_STRATEGY_PER_LANG
            )
            available_for_this_verdict = global_verdict_availability.get(verdict, 0)
            target_for_this_verdict = 0
            if available_for_this_verdict == 0:
                target_samples_per_verdict_for_lang[verdict] = 0
                continue
            if config["strategy"] == "min_samples_or_all":
                target_for_this_verdict = min(
                    config["value"], available_for_this_verdict
                )
            elif config["strategy"] == "max_samples":
                target_for_this_verdict = min(
                    config["value"], available_for_this_verdict
                )
            elif (
                config["strategy"] == "target_proportion_of_max_class"
                and max_class_count > 0
            ):
                target_for_this_verdict = int(config["value"] * max_class_count)
                target_for_this_verdict = min(
                    target_for_this_verdict, available_for_this_verdict
                )
                if "cap_at" in config:
                    target_for_this_verdict = min(
                        target_for_this_verdict, config["cap_at"]
                    )
            elif config["strategy"] == "as_is":
                target_for_this_verdict = available_for_this_verdict
            target_samples_per_verdict_for_lang[verdict] = target_for_this_verdict

        print(f"Global sampling targets for {lang_name_key}:")
        for verdict, count in target_samples_per_verdict_for_lang.items():
            print(
                f"  {verdict}: {count} (Available: {global_verdict_availability.get(verdict, 0)})"
            )

        current_total_targeted = sum(target_samples_per_verdict_for_lang.values())
        if (
            MAX_TOTAL_SAMPLES_PER_LANGUAGE
            and current_total_targeted > MAX_TOTAL_SAMPLES_PER_LANGUAGE
        ):
            print(
                f"Total targeted ({current_total_targeted}) exceeds MAX_TOTAL_SAMPLES ({MAX_TOTAL_SAMPLES_PER_LANGUAGE}). Scaling down."
            )
            scale_factor = MAX_TOTAL_SAMPLES_PER_LANGUAGE / current_total_targeted
            for v_key in target_samples_per_verdict_for_lang:
                target_samples_per_verdict_for_lang[v_key] = int(
                    target_samples_per_verdict_for_lang[v_key] * scale_factor
                )
            print(f"Scaled global sampling targets for {lang_name_key}:")
            for v, c in target_samples_per_verdict_for_lang.items():
                print(f"  {v}: {c}")

        # 3. Second Pass: Collect submissions
        collected_code_data: List[Dict] = []
        collected_stats_data: List[Dict] = []
        indices_per_verdict: Dict[str, list] = {v: [] for v in GLOBAL_TARGET_VERDICTS}
        for index, row in lang_specific_meta_df.iterrows():
            indices_per_verdict[row["target_verdict"]].append(index)
        for verdict in indices_per_verdict:
            random.shuffle(indices_per_verdict[verdict])

        final_selected_indices = []
        for verdict, target_count in target_samples_per_verdict_for_lang.items():
            available_indices_for_verdict = indices_per_verdict.get(verdict, [])
            final_selected_indices.extend(available_indices_for_verdict[:target_count])

        if not final_selected_indices:
            print(
                f"No samples selected for {lang_name_key} based on global targets. Skipping file processing."
            )
            continue

        sampled_lang_df = lang_specific_meta_df.loc[final_selected_indices].copy()
        sampled_lang_df = sampled_lang_df.sample(
            frac=1, random_state=RANDOM_SEED
        ).reset_index(drop=True)
        print(
            f"Globally sampled {len(sampled_lang_df)} submissions for {lang_name_key} to process."
        )

        # --- WRAP THIS LOOP WITH TQDM ---
        for _, submission_row in tqdm(
            sampled_lang_df.iterrows(),
            total=len(sampled_lang_df),
            desc=f"Extracting files for {lang_name_key}",
        ):
            p_id, s_id, s_fname_ext = (
                submission_row["problem_id"],
                submission_row["submission_id"],
                submission_row["filename_ext"],
            )
            s_fname_full = str(s_id) + "." + str(s_fname_ext)
            verdict_category = submission_row["target_verdict"]

            f_path = os.path.join(
                CODENET_DATA_DIR, str(p_id), codenet_data_subdir, s_fname_full
            )

            try:
                normalized_code = code_normalizer_instance.normalize_file(
                    f_path, lang=lang_name_key
                )
                if normalized_code is None or not normalized_code.strip():
                    continue

                collected_code_data.append(
                    {
                        "problem_id": str(p_id),
                        "submission_file": s_fname_full,
                        "code": normalized_code,
                    }
                )
                collected_stats_data.append(
                    {
                        "problem_id": str(p_id),
                        "submission_id": str(s_id),
                        "language": lang_name_key,
                        "verdict": verdict_category,
                        "runtime": submission_row["cpu_time"],
                        "memory": submission_row["memory"],
                        "code_size": submission_row["code_size"],
                    }
                )
            except FileNotFoundError:
                pass
            except ValueError:
                pass
            except Exception as e:
                print(
                    f"Warning: Unexpected error processing {f_path}: {type(e).__name__} - {e}"
                )
        # --- END OF TQDM WRAPPED LOOP ---

        if collected_code_data:
            df_c = pd.DataFrame(collected_code_data)
            df_c.to_csv(code_output_csv, index=False, encoding="utf-8")
            print(f"Saved {len(df_c)} code entries to {code_output_csv}")
        else:
            print(f"No code entries collected for {lang_name_key}.")

        if collected_stats_data:
            df_s = pd.DataFrame(collected_stats_data)
            df_s_counts = Counter(df_s["verdict"])
            print(f"Final verdict distribution for {lang_name_key} in stats file:")
            for v, c in df_s_counts.items():
                print(f"  {v}: {c}")
            df_s.to_csv(stats_output_csv, index=False, encoding="utf-8")
            print(f"Saved {len(df_s)} stats entries to {stats_output_csv}")
        else:
            print(f"No stats entries collected for {lang_name_key}.")

    print("\nAll processing finished.")


if __name__ == "__main__":
    main()
