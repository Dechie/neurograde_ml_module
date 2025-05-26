import os
import pandas as pd
import shutil
from tqdm import tqdm
from typing import Dict, List, Set

# --- Configuration ---
INPUT_ANALYSIS_CSV = "samples.csv"  # CSV from previous script
OUTPUT_SHOWCASE_DIR = "showcase_problem_submissions/"  # Base output directory

# --- Paths from your extract.py (needed to find original files) ---
PROJECT_CODENET_BASE_DIR = "../Project_CodeNet/"  # IMPORTANT: Adjust if different
CODENET_PROBLEM_DESCRIPTIONS_DIR = os.path.join(
    PROJECT_CODENET_BASE_DIR, "problem_descriptions"
)
CODENET_DATA_DIR = os.path.join(PROJECT_CODENET_BASE_DIR, "data")

# --- Language mapping and info (similar to extract.py) ---
# This maps the language string used in your CSV (e.g., 'python')
# to the CodeNet directory name and the typical file extension.
LANGUAGE_DETAILS_FOR_SHOWCASE = {
    "cpp": {"codenet_dir_name": "C++", "extension": "cpp"},
    "python": {"codenet_dir_name": "Python", "extension": "py"},
    "java": {"codenet_dir_name": "Java", "extension": "java"},
    # Add other languages if they are in your INPUT_ANALYSIS_CSV
}

# How many unique problems to collect showcases for
MAX_PROBLEMS_FOR_SHOWCASE = 20
# How many submissions per verdict per language per problem (from your previous request)
# This script will just copy what's in the input CSV for the selected problems.
# The diversity was aimed for when creating INPUT_ANALYSIS_CSV.


# --- Main Script ---
def create_showcase_folder():
    print(f"--- Starting Showcase Collection ---")
    print(f"Reading analyzed samples from: {INPUT_ANALYSIS_CSV}")

    if not os.path.exists(INPUT_ANALYSIS_CSV):
        print(
            f"ERROR: Input CSV '{INPUT_ANALYSIS_CSV}' not found. Please run the previous analysis script first."
        )
        return

    try:
        df_analyzed = pd.read_csv(INPUT_ANALYSIS_CSV)
        if df_analyzed.empty:
            print(f"Input CSV '{INPUT_ANALYSIS_CSV}' is empty. No samples to process.")
            return
    except Exception as e:
        print(f"Error reading input CSV '{INPUT_ANALYSIS_CSV}': {e}")
        return

    # Ensure required columns are present
    required_cols = [
        "problem_id",
        "submission_id",
        "language_model_used",
        "true_verdict_string",
    ]
    for col in required_cols:
        if col not in df_analyzed.columns:
            print(
                f"ERROR: Required column '{col}' not found in '{INPUT_ANALYSIS_CSV}'."
            )
            return

    os.makedirs(OUTPUT_SHOWCASE_DIR, exist_ok=True)
    print(f"Output will be saved in: {OUTPUT_SHOWCASE_DIR}")

    collected_problem_ids: Set[str] = set()
    problems_processed_count = 0

    # Iterate through problems found in the analyzed CSV
    # Group by problem_id to process one problem at a time
    for problem_id, group_df in tqdm(
        df_analyzed.groupby("problem_id"), desc="Processing Problems"
    ):
        if problems_processed_count >= MAX_PROBLEMS_FOR_SHOWCASE:
            print(
                f"Collected showcase for {MAX_PROBLEMS_FOR_SHOWCASE} problems. Stopping."
            )
            break

        problem_id_str = str(problem_id)  # Ensure it's a string
        problem_output_dir = os.path.join(OUTPUT_SHOWCASE_DIR, problem_id_str)

        # --- 1. Copy Problem Statement (HTML) ---
        original_statement_html_path = os.path.join(
            CODENET_PROBLEM_DESCRIPTIONS_DIR, f"{problem_id_str}.html"
        )

        if os.path.exists(original_statement_html_path):
            os.makedirs(
                problem_output_dir, exist_ok=True
            )  # Create problem subfolder only if statement exists
            try:
                shutil.copy2(
                    original_statement_html_path,
                    os.path.join(
                        problem_output_dir, f"{problem_id_str}_statement.html"
                    ),
                )
                # print(f"  Copied statement for problem {problem_id_str}")
            except Exception as e:
                print(
                    f"  Warning: Could not copy statement for problem {problem_id_str}: {e}"
                )
                # If statement can't be copied, maybe skip this problem or just its statement
                continue  # Let's skip problem if statement is missing, or it won't be very useful
        else:
            print(
                f"  Warning: Original HTML statement not found for problem {problem_id_str} at '{original_statement_html_path}'. Skipping this problem."
            )
            continue  # Skip this problem if no HTML statement

        # --- 2. Copy Submission Files for this Problem ---
        submissions_copied_for_this_problem = 0
        for _, row in group_df.iterrows():
            submission_id = str(row["submission_id"])
            language_key = str(
                row["language_model_used"]
            ).lower()  # e.g., "python", "cpp"
            true_verdict = str(row["true_verdict_string"]).replace(
                " ", "_"
            )  # Sanitize for filename

            if language_key not in LANGUAGE_DETAILS_FOR_SHOWCASE:
                print(
                    f"    Warning: Language '{language_key}' for submission {submission_id} (problem {problem_id_str}) not configured in LANGUAGE_DETAILS_FOR_SHOWCASE. Skipping this submission."
                )
                continue

            lang_details = LANGUAGE_DETAILS_FOR_SHOWCASE[language_key]
            codenet_lang_dir = lang_details["codenet_dir_name"]  # e.g., "C++", "Python"
            file_extension = lang_details["extension"]  # e.g., "cpp", "py"

            # Construct the original submission filename (submission_id should be filename root)
            original_submission_filename = f"{submission_id}.{file_extension}"
            original_submission_path = os.path.join(
                CODENET_DATA_DIR,
                problem_id_str,
                codenet_lang_dir,
                original_submission_filename,
            )

            # Construct the new filename: {submission_id}_{true_verdict}.{extension}
            new_submission_filename = f"{submission_id}_{true_verdict}.{file_extension}"
            new_submission_path = os.path.join(
                problem_output_dir, new_submission_filename
            )

            if os.path.exists(original_submission_path):
                try:
                    shutil.copy2(original_submission_path, new_submission_path)
                    submissions_copied_for_this_problem += 1
                except Exception as e:
                    print(
                        f"    Warning: Could not copy submission {original_submission_filename} for problem {problem_id_str}: {e}"
                    )
            else:
                # This might happen if the INPUT_ANALYSIS_CSV references submissions that
                # were not found or skipped during your initial `extract.py` run,
                # or if `submission_id` in CSV doesn't match filename root.
                print(
                    f"    Warning: Original submission file not found for s_id {submission_id} (problem {problem_id_str}) at '{original_submission_path}'. Skipping this submission."
                )

        if submissions_copied_for_this_problem > 0:
            print(
                f"  For Problem {problem_id_str}: Copied statement and {submissions_copied_for_this_problem} submission files."
            )
            collected_problem_ids.add(
                problem_id_str
            )  # Mark this problem as processed for showcase
            problems_processed_count = len(collected_problem_ids)
        else:
            # If we copied the statement but no submissions, we might want to remove the empty problem folder
            # or just leave it with only the statement. Let's leave it for now.
            print(
                f"  For Problem {problem_id_str}: Copied statement, but found no corresponding original submission files to copy based on input CSV."
            )

    print(f"\n--- Showcase Collection Finished ---")
    print(
        f"Collected showcases for {len(collected_problem_ids)} unique problems in '{OUTPUT_SHOWCASE_DIR}'."
    )


if __name__ == "__main__":
    # Basic checks before starting
    if not os.path.exists(INPUT_ANALYSIS_CSV):
        print(f"ERROR: Input analysis CSV '{INPUT_ANALYSIS_CSV}' not found.")
        exit()
    if not os.path.isdir(PROJECT_CODENET_BASE_DIR):
        print(
            f"ERROR: Project CodeNet base directory '{PROJECT_CODENET_BASE_DIR}' not found. Please check path."
        )
        exit()
    if not os.path.isdir(CODENET_PROBLEM_DESCRIPTIONS_DIR):
        print(
            f"ERROR: CodeNet problem descriptions directory '{CODENET_PROBLEM_DESCRIPTIONS_DIR}' not found."
        )
        exit()
    if not os.path.isdir(CODENET_DATA_DIR):
        print(f"ERROR: CodeNet data directory '{CODENET_DATA_DIR}' not found.")
        exit()

    create_showcase_folder()
