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

GLOBAL_TARGET_VERDICTS = [  # Used for initial filtering and as the universe of verdicts
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

# --- NEW SAMPLING PARAMETER ---
TARGET_SUBMISSIONS_PER_PROBLEM_LANG = 160


# --- Helper Functions --- (Keep these as they are)
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


def get_stratified_sample_counts(
    verdict_counts: dict, total_target_samples: int
) -> dict:
    """
    Calculates the number of samples for each verdict category using stratified sampling
    to meet a total target, considering availability.
    Args:
        verdict_counts: Dict of {verdict_name: count_available}.
        total_target_samples: The desired total number of samples across all verdicts.
    Returns:
        Dict of {verdict_name: num_to_sample}.
    """
    # Filter out verdicts with zero counts to avoid division by zero
    # and to only consider strata that actually exist.
    active_verdict_counts = {v: c for v, c in verdict_counts.items() if c > 0}
    if not active_verdict_counts:
        return {
            v: 0 for v in verdict_counts.keys()
        }  # Return all zeros if no active verdicts

    total_available_for_active_verdicts = sum(active_verdict_counts.values())

    # If total available is less than or equal to the target, take all available
    if total_available_for_active_verdicts <= total_target_samples:
        return active_verdict_counts.copy()  # Take all available from active verdicts

    # Calculate ideal (float) number of samples based on proportion
    ideal_samples = {}
    for verdict, count in active_verdict_counts.items():
        proportion = count / total_available_for_active_verdicts
        ideal_samples[verdict] = proportion * total_target_samples

    # Allocate samples: initial allocation is floor, then distribute remainders
    # based on largest fractional parts, while respecting availability.

    allocated_samples = {v: 0 for v in active_verdict_counts.keys()}

    # Phase 1: Hamilton method / largest remainder method for initial distribution
    # Calculate initial integer allocations (floor) and remainders
    initial_allocations = {v: int(s) for v, s in ideal_samples.items()}
    remainders = {v: s - initial_allocations[v] for v, s in ideal_samples.items()}

    current_total_allocated = sum(initial_allocations.values())
    num_remaining_to_allocate = total_target_samples - current_total_allocated

    # Sort verdicts by their remainders in descending order to prioritize
    sorted_verdicts_by_remainder = sorted(
        remainders.keys(), key=lambda v: remainders[v], reverse=True
    )

    for verdict in sorted_verdicts_by_remainder:
        if num_remaining_to_allocate <= 0:
            break
        # Can only increment if we don't exceed available for this verdict
        if initial_allocations[verdict] < active_verdict_counts[verdict]:
            initial_allocations[verdict] += 1
            num_remaining_to_allocate -= 1

    # Phase 2: Adjust allocations to not exceed available counts
    # This step is crucial. After proportional allocation, ensure no category asks for more than it has.
    # Redistribute any "excess" demand.

    final_samples_to_take = {}
    current_total_taken = 0

    # Tentatively take the proportionally allocated amounts, capped by availability
    for verdict in active_verdict_counts.keys():  # Iterate in a consistent order
        final_samples_to_take[verdict] = min(
            initial_allocations.get(verdict, 0), active_verdict_counts[verdict]
        )
        current_total_taken += final_samples_to_take[verdict]

    # If we are short of the target due to capping by availability, try to top up from
    # other verdicts that have more available than their current allocation.
    if current_total_taken < total_target_samples:
        shortfall = total_target_samples - current_total_taken
        # Sort by who has the most "room" (available - current_allocation)
        # and hasn't hit their ideal proportional share yet (or has more available)

        # Create a list of (verdict, room_to_add)
        potential_top_up_sources = []
        for verdict in active_verdict_counts.keys():
            room = active_verdict_counts[verdict] - final_samples_to_take[verdict]
            if room > 0:
                potential_top_up_sources.append((verdict, room))

        # Sort by room (descending) to prioritize those with more capacity
        potential_top_up_sources.sort(key=lambda x: x[1], reverse=True)

        for verdict, room in potential_top_up_sources:
            if shortfall <= 0:
                break
            can_add = min(shortfall, room)
            final_samples_to_take[verdict] += can_add
            shortfall -= can_add
            current_total_taken += (
                can_add  # Not strictly needed here as shortfall tracks
            )

    # Ensure all original GLOBAL_TARGET_VERDICTS keys are present, with 0 if not active
    result = {v: 0 for v in verdict_counts.keys()}
    result.update(final_samples_to_take)
    return result


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
    print("Loading CodeNet metadata...")
    for p_id in tqdm(user_problem_ids, desc="Loading metadata per problem"):
        fpath = os.path.join(CODENET_METADATA_PER_PROBLEM_DIR, f"{p_id}.csv")
        try:
            all_problem_metadata_dfs.append(pd.read_csv(fpath, low_memory=False))
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Warning: Error reading {fpath}: {e}")

    if not all_problem_metadata_dfs:
        print("No CodeNet metadata loaded. Exiting.")
        return
    codenet_full_meta_df = pd.concat(all_problem_metadata_dfs, ignore_index=True)
    print(
        f"Loaded metadata for {codenet_full_meta_df['problem_id'].nunique()} problems, {len(codenet_full_meta_df)} entries."
    )

    print("Preprocessing CodeNet metadata...")
    codenet_full_meta_df["target_language"] = codenet_full_meta_df["language"].apply(
        map_codenet_language_to_target
    )
    codenet_full_meta_df["target_verdict"] = codenet_full_meta_df["status"].apply(
        map_codenet_status_to_target_verdict
    )

    codenet_processed_meta_df = codenet_full_meta_df.dropna(
        subset=METADATA_REQUIRED_COLS + ["target_language", "target_verdict"]
    ).copy()
    codenet_processed_meta_df = codenet_processed_meta_df[
        codenet_processed_meta_df["target_verdict"].isin(GLOBAL_TARGET_VERDICTS)
    ]
    print(
        f"Filtered to {len(codenet_processed_meta_df)} relevant CodeNet metadata entries."
    )

    if codenet_processed_meta_df.empty:
        print("No relevant CodeNet submissions found. Exiting.")
        return

    for lang_name_key, lang_details in TARGET_LANGUAGES_INFO.items():
        target_lang_for_normalizer = lang_name_key
        code_output_csv, stats_output_csv = (
            lang_details["code_csv"],
            lang_details["stats_csv"],
        )
        codenet_data_subdir = lang_details["codenet_dir_name"]

        print(f"\n--- Processing language: {target_lang_for_normalizer} ---")
        collected_code_data, collected_stats_data = [], []

        lang_specific_meta_df = codenet_processed_meta_df[
            codenet_processed_meta_df["target_language"] == target_lang_for_normalizer
        ].copy()
        if lang_specific_meta_df.empty:
            print(f"No submissions for '{target_lang_for_normalizer}'. Skipping.")
            continue

        for p_id in tqdm(
            user_problem_ids, desc=f"Problems for {target_lang_for_normalizer}"
        ):
            problem_lang_submissions_df = lang_specific_meta_df[
                lang_specific_meta_df["problem_id"] == p_id
            ]
            if problem_lang_submissions_df.empty:
                continue

            # --- STRATIFIED SAMPLING LOGIC ---
            # 1. Get current counts of each verdict for this problem/language
            current_verdict_counts = {}
            for (
                v_cat
            ) in (
                GLOBAL_TARGET_VERDICTS
            ):  # Ensure all global verdicts are considered for counts dict
                current_verdict_counts[v_cat] = len(
                    problem_lang_submissions_df[
                        problem_lang_submissions_df["target_verdict"] == v_cat
                    ]
                )

            # 2. Calculate how many to sample from each verdict to reach TARGET_SUBMISSIONS_PER_PROBLEM_LANG
            # The function will handle cases where total available < target.
            num_to_sample_per_verdict = get_stratified_sample_counts(
                current_verdict_counts, TARGET_SUBMISSIONS_PER_PROBLEM_LANG
            )
            # --- END OF STRATIFIED SAMPLING LOGIC ---

            total_sampled_for_problem = 0  # For verification
            for verdict_category, num_to_sample in num_to_sample_per_verdict.items():
                if num_to_sample == 0:
                    continue

                verdict_specific_df = problem_lang_submissions_df[
                    problem_lang_submissions_df["target_verdict"] == verdict_category
                ]

                # Ensure we don't try to sample more than available (should be handled by get_stratified_sample_counts)
                actual_num_to_sample = min(num_to_sample, len(verdict_specific_df))

                if actual_num_to_sample > 0:
                    sampled_df = verdict_specific_df.sample(
                        n=actual_num_to_sample, random_state=RANDOM_SEED
                    )
                    total_sampled_for_problem += len(sampled_df)

                    for _, submission_row in sampled_df.iterrows():
                        s_id, s_fname = (
                            submission_row["submission_id"],
                            submission_row["submission_id"]
                            + "."
                            + submission_row["filename_ext"],
                        )

                        f_path = os.path.join(
                            CODENET_DATA_DIR,
                            p_id,
                            codenet_data_subdir,
                            s_fname,
                        )

                        try:
                            norm_code = CodeNormalizer.normalize_file(f_path)
                            collected_code_data.append(
                                {
                                    "problem_id": p_id,
                                    "submission_file": s_fname,
                                    "code": norm_code,
                                }
                            )
                            collected_stats_data.append(
                                {
                                    "problem_id": p_id,
                                    "submission_id": s_id,
                                    "language": target_lang_for_normalizer,
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
                            print(f"Warning: Error processing {f_path}: {e}")

            # print(
            #     f"Problem {p_id} ({target_lang_for_normalizer}): Target {TARGET_SUBMISSIONS_PER_PROBLEM_LANG}, Actual sampled: {total_sampled_for_problem}"
            # )

        if collected_code_data:
            pd.DataFrame(collected_code_data).to_csv(
                code_output_csv, index=False, encoding="utf-8"
            )
            print(f"Saved {len(collected_code_data)} code entries to {code_output_csv}")
        else:
            print(f"No code entries for {target_lang_for_normalizer}.")

        if collected_stats_data:
            pd.DataFrame(collected_stats_data).to_csv(
                stats_output_csv, index=False, encoding="utf-8"
            )
            print(
                f"Saved {len(collected_stats_data)} stats entries to {stats_output_csv}"
            )
        else:
            print(f"No stats entries for {target_lang_for_normalizer}.")

    print("\nAll processing finished.")


if __name__ == "__main__":
    main()
