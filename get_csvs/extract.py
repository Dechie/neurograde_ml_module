import pandas as pd
import os
import random
from tqdm import tqdm
from code_normalizer import CodeNormalizer

# --- Configuration ---
PROBLEM_STATEMENTS_CSV = "data/final_problem_statements.csv"
PROJECT_CODENET_BASE_DIR = "../Project_CodeNet/"
CODENET_METADATA_PER_PROBLEM_DIR = os.path.join(PROJECT_CODENET_BASE_DIR, "metadata")
CODENET_DATA_DIR = os.path.join(PROJECT_CODENET_BASE_DIR, "data")

OUTPUT_DIR = "data/"
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

TARGET_VERDICTS_ORDERED = [
    "Accepted",
    "Compile Error",
    "Memory Limit Exceeded",
    "Presentation Error",
    "Runtime Error",
    "Time Limit Exceeded",
    "Wrong Answer",
]
SUBMISSIONS_PER_VERDICT_PER_PROBLEM = 25
RANDOM_SEED = 42

# Expected columns in the per-problem metadata CSVs (based on your description)
# submission_id,problem_id,user_id,date,language,original_language,filename_ext,status,cpu_time,memory,code_size,accuracy
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
        return "Runtime Error"
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
            print(f"No problem_ids found in {PROBLEM_STATEMENTS_CSV}. Exiting.")
            return
        print(
            f"Loaded {len(user_problem_ids)} unique problem IDs from {PROBLEM_STATEMENTS_CSV}"
        )
    except Exception as e:
        print(f"Error loading {PROBLEM_STATEMENTS_CSV}: {e}")
        return

    all_problem_metadata_dfs = []
    print("Loading CodeNet metadata from individual problem CSV files...")
    for p_id in tqdm(user_problem_ids, desc="Loading metadata per problem"):
        individual_meta_file_path = os.path.join(
            CODENET_METADATA_PER_PROBLEM_DIR, f"{p_id}.csv"
        )
        try:
            temp_df = pd.read_csv(individual_meta_file_path, low_memory=False)
            # Ensure all required columns are present for safety, though problem_id should be.
            # If problem_id is missing in a file, it's a data issue, but we'll rely on it being there.
            all_problem_metadata_dfs.append(temp_df)
        except FileNotFoundError:
            # print(f"Warning: Metadata file not found: {individual_meta_file_path}")
            pass
        except Exception as e:
            print(f"Warning: Error reading metadata {individual_meta_file_path}: {e}")

    if not all_problem_metadata_dfs:
        print("No CodeNet metadata could be loaded. Exiting.")
        return

    codenet_full_meta_df = pd.concat(all_problem_metadata_dfs, ignore_index=True)
    print(
        f"Loaded metadata for {codenet_full_meta_df['problem_id'].nunique()} problems, total {len(codenet_full_meta_df)} entries."
    )

    print("Preprocessing CodeNet metadata...")
    codenet_full_meta_df["target_language"] = codenet_full_meta_df["language"].apply(
        map_codenet_language_to_target
    )
    codenet_full_meta_df["target_verdict"] = codenet_full_meta_df["status"].apply(
        map_codenet_status_to_target_verdict
    )

    # Drop rows where essential data for filtering or output is missing
    codenet_processed_meta_df = codenet_full_meta_df.dropna(
        subset=METADATA_REQUIRED_COLS + ["target_language", "target_verdict"]
    ).copy()

    # Filter for entries that successfully mapped to one of the 7 target verdicts
    codenet_processed_meta_df = codenet_processed_meta_df[
        codenet_processed_meta_df["target_verdict"].isin(TARGET_VERDICTS_ORDERED)
    ]
    print(
        f"Filtered to {len(codenet_processed_meta_df)} relevant CodeNet metadata entries after mapping and cleaning."
    )

    if codenet_processed_meta_df.empty:
        print("No relevant CodeNet submissions found. Exiting.")
        return

    for lang_name_key, lang_details in TARGET_LANGUAGES_INFO.items():
        target_lang_for_normalizer = lang_name_key  # "C++", "Python", "Java"
        code_output_csv = lang_details["code_csv"]
        stats_output_csv = lang_details["stats_csv"]
        codenet_data_subdir = lang_details[
            "codenet_dir_name"
        ]  # "CPP", "Python", "Java"

        print(f"\n--- Processing language: {target_lang_for_normalizer} ---")

        collected_code_data = []
        collected_stats_data = []

        lang_specific_meta_df = codenet_processed_meta_df[
            codenet_processed_meta_df["target_language"] == target_lang_for_normalizer
        ].copy()

        if lang_specific_meta_df.empty:
            print(
                f"No submissions found for language '{target_lang_for_normalizer}'. Skipping."
            )
            continue

        for p_id in tqdm(
            user_problem_ids, desc=f"Problems for {target_lang_for_normalizer}"
        ):
            problem_lang_submissions_df = lang_specific_meta_df[
                lang_specific_meta_df["problem_id"] == p_id
            ]
            if problem_lang_submissions_df.empty:
                continue

            for verdict_category in TARGET_VERDICTS_ORDERED:
                verdict_specific_df = problem_lang_submissions_df[
                    problem_lang_submissions_df["target_verdict"] == verdict_category
                ]
                if verdict_specific_df.empty:
                    continue

                num_to_sample = min(
                    len(verdict_specific_df), SUBMISSIONS_PER_VERDICT_PER_PROBLEM
                )
                sampled_df = verdict_specific_df.sample(
                    n=num_to_sample, random_state=RANDOM_SEED
                )

                for _, submission_row in sampled_df.iterrows():
                    submission_id = submission_row["submission_id"]
                    # Use 'filename_ext' for the submission file name and constructing path
                    submission_filename = (
                        submission_id + "." + submission_row["filename_ext"]
                    )

                    file_path = os.path.join(
                        CODENET_DATA_DIR, p_id, codenet_data_subdir, submission_filename
                    )

                    try:
                        normalized_code = CodeNormalizer.normalize_file(file_path)

                        # Data for submissions_{Language}.csv
                        collected_code_data.append(
                            {
                                "problem_id": p_id,
                                "submission_file": submission_filename,  # Using filename_ext
                                "code": normalized_code,
                            }
                        )

                        # Data for submission_stats_{Language}.csv
                        collected_stats_data.append(
                            {
                                "problem_id": p_id,
                                "submission_id": submission_id,
                                "language": target_lang_for_normalizer,  # The mapped "C++", "Python", "Java"
                                "verdict": verdict_category,
                                "runtime": submission_row["cpu_time"],
                                "memory": submission_row["memory"],
                                "code_size": submission_row["code_size"],
                            }
                        )
                    except FileNotFoundError:
                        # print(f"Warning: Code file not found: {file_path}")
                        pass
                    except ValueError as ve:  # From CodeNormalizer if ext not supported
                        # print(f"Warning: Normalization error for {file_path}: {ve}")
                        pass
                    except Exception as e:
                        print(
                            f"Warning: Error processing/normalizing file {file_path}: {e}"
                        )

        # Save code submissions CSV
        if collected_code_data:
            code_df = pd.DataFrame(collected_code_data)
            try:
                code_df.to_csv(code_output_csv, index=False, encoding="utf-8")
                print(f"Saved {len(code_df)} code entries to {code_output_csv}")
            except Exception as e:
                print(f"Error saving {code_output_csv}: {e}")
        else:
            print(f"No code entries collected for {target_lang_for_normalizer}.")

        # Save stats submissions CSV
        if collected_stats_data:
            stats_df = pd.DataFrame(collected_stats_data)
            try:
                stats_df.to_csv(stats_output_csv, index=False, encoding="utf-8")
                print(f"Saved {len(stats_df)} stats entries to {stats_output_csv}")
            except Exception as e:
                print(f"Error saving {stats_output_csv}: {e}")
        else:
            print(f"No stats entries collected for {target_lang_for_normalizer}.")

    print("\nAll processing finished.")


if __name__ == "__main__":
    main()
