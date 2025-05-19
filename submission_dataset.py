# submission_dataset.py

import os
import pandas as pd
import torch
from torch.utils.data import (
    Dataset,
)  # DataLoader removed from here, will be in train_model.py
from typing import Dict, Tuple, List, Any, Optional

# --- Import Real Components --- (Ensure these .py files exist)
try:
    from preprocessor import Preprocessor
    from code_normalizer import CodeNormalizer

    # ConcatEmbedder is NOT directly used by SubmissionDataset anymore.
    # It's used by SubmissionPredictor.
    # The dataset will only use Preprocessor and CodeNormalizer.
except ImportError as e:
    print(
        f"ERROR (submission_dataset.py): Failed to import Preprocessor or CodeNormalizer: {e}"
    )

    # Define dummy classes if imports fail, for basic script parsing
    class Preprocessor:  # type: ignore
        def preprocess_text(self, text: str) -> str:
            return text

    class CodeNormalizer:  # type: ignore
        def normalize_code(self, code: str, lang: str) -> str:
            return code


# --- End of Imports ---


class SubmissionDataset(Dataset):
    def __init__(
        self,
        stats_csv_path: str,
        code_csv_path: str,
        problem_csv_path: str,
        preprocessor_instance: Preprocessor,
        code_normalizer_instance: CodeNormalizer,
        # ConcatEmbedder is NO LONGER passed to SubmissionDataset
    ):
        if not isinstance(preprocessor_instance, Preprocessor):
            raise TypeError(
                "preprocessor_instance must be an instance of Preprocessor."
            )
        if not isinstance(code_normalizer_instance, CodeNormalizer):
            raise TypeError(
                "code_normalizer_instance must be an instance of CodeNormalizer."
            )

        self.preprocessor = preprocessor_instance
        self.code_normalizer = code_normalizer_instance

        # Load problem statements
        try:
            df_probs = pd.read_csv(problem_csv_path, dtype={"problem_id": str})
        except FileNotFoundError:
            print(f"Error: Problem statements file not found at {problem_csv_path}")
            raise

        df_probs["raw_statement"] = (
            df_probs["statement"].fillna("").astype(str).str.strip()
            + "\nInput: "
            + df_probs["input_spec"].fillna("").astype(str).str.strip()
            + "\nOutput: "
            + df_probs["output_spec"].fillna("").astype(str).str.strip()
        )
        # Preprocess statements here for consistency before they are potentially embedded
        df_probs["processed_statement"] = df_probs["raw_statement"].apply(
            self.preprocessor.preprocess_text
        )
        self.problem_statements: Dict[str, str] = df_probs.set_index("problem_id")[
            "processed_statement"  # Store PREPROCESSED statements
        ].to_dict()

        # Load submission stats and code
        try:
            df_stats = pd.read_csv(
                stats_csv_path, dtype={"problem_id": str, "submission_id": str}
            )
            df_code = pd.read_csv(
                code_csv_path, dtype={"problem_id": str, "submission_file": str}
            )
        except FileNotFoundError:
            print(
                f"Error: Stats or Code CSV not found. Searched for: '{stats_csv_path}', '{code_csv_path}'"
            )
            raise

        if "language" in df_stats.columns:
            unique_langs = (
                df_stats["language"].dropna().astype(str).str.lower().unique()
            )
            if len(unique_langs) == 1:
                self.lang = unique_langs[
                    0
                ]  # This lang applies to all items from this dataset instance
            elif not unique_langs.size:
                print(
                    f"Warning: 'language' column in {stats_csv_path} is empty or all NaN. Inferring lang from filename."
                )
                self.lang = (
                    os.path.basename(stats_csv_path).split("_")[2].split(".")[0].lower()
                )
            else:
                raise ValueError(
                    f"Stats file {stats_csv_path} contains multiple languages: {unique_langs}. Dataset expects one uniform language."
                )
        else:
            print(
                f"Warning: 'language' column not in {stats_csv_path}. Inferring lang from filename."
            )
            self.lang = (
                os.path.basename(stats_csv_path).split("_")[2].split(".")[0].lower()
            )
        print(
            f"Info (SubmissionDataset): Determined language as '{self.lang}' for dataset based on '{os.path.basename(stats_csv_path)}'"
        )

        df_code = df_code.rename(columns={"submission_file": "submission_id"})
        df_code["submission_id"] = df_code["submission_id"].apply(
            lambda fn: os.path.splitext(str(fn))[0]
        )
        df_code["submission_id"] = df_code["submission_id"].astype(str)

        self.df = pd.merge(
            df_stats, df_code, on=["problem_id", "submission_id"], how="inner"
        )
        original_rows_before_problem_filter = len(self.df)
        self.df = self.df[self.df["problem_id"].isin(self.problem_statements.keys())]
        if len(self.df) < original_rows_before_problem_filter:
            print(
                f"Info: Dropped {original_rows_before_problem_filter - len(self.df)} rows due to missing problem_id in problem statements."
            )

        essential_cols = [
            "code",
            "verdict",
            "runtime",
            "memory",
        ]  # Add other target columns if they become mandatory
        original_rows_before_na_drop = len(self.df)
        self.df.dropna(subset=essential_cols, inplace=True)
        if len(self.df) < original_rows_before_na_drop:
            print(
                f"Info: Dropped {original_rows_before_na_drop - len(self.df)} rows due to missing data in {essential_cols}."
            )

        self.df = self.df.reset_index(drop=True)

        if len(self.df) == 0:
            print(
                f"Warning (SubmissionDataset): Dataset is EMPTY after loading and merging data for lang '{self.lang}'."
            )
        else:
            print(
                f"Info (SubmissionDataset): Loaded {len(self.df)} samples for language '{self.lang}'."
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # Now returns a dictionary
        if idx >= len(self.df):
            raise IndexError("Index out of bounds for SubmissionDataset.")

        row = self.df.iloc[idx]
        pid = str(row["problem_id"])

        raw_code = str(row["code"]) if pd.notna(row["code"]) else ""
        # Code normalization happens here
        normalized_code = self.code_normalizer.normalize_code(raw_code, self.lang)

        # Statement is already preprocessed (during __init__) and fetched from self.problem_statements
        processed_stmt = self.problem_statements.get(pid)
        if processed_stmt is None:
            print(
                f"Warning (getitem): problem_id '{pid}' not found in preloaded statements despite filtering. Using empty statement."
            )
            processed_stmt = ""

        verdict_str = str(
            row["verdict"]
        )  # Keep original verdict string for potential debugging/logging
        verdict_encoded = self._encode_verdict(verdict_str)
        if (
            verdict_encoded == -1 and verdict_str
        ):  # Log only if verdict_str was not empty
            print(
                f"Warning (getitem): Unmapped verdict string '{verdict_str}' for idx {idx}, pid {pid}."
            )

        # Return a dictionary of raw/processed inputs and tensorized targets
        return {
            "code_str": normalized_code,  # For CodeEmbedderGNN (via ConcatEmbedder)
            "statement_str": processed_stmt,  # For TextEmbedder (via ConcatEmbedder)
            "lang_str": self.lang,  # For ConcatEmbedder
            "verdict_raw_str": verdict_str,  # For inspection/debugging
            "verdict_encoded": torch.tensor(verdict_encoded, dtype=torch.long),
            "runtime": torch.tensor(float(row["runtime"]), dtype=torch.float),
            "memory": torch.tensor(float(row["memory"]), dtype=torch.float),
        }

    @staticmethod
    def _encode_verdict(verdict_str: str) -> int:
        normalized_verdict = (
            str(verdict_str).strip().title() if pd.notna(verdict_str) else ""
        )
        mapping = {
            "Accepted": 0,
            "Wrong Answer": 1,
            "Time Limit Exceeded": 2,
            "Memory Limit Exceeded": 3,
            "Runtime Error": 4,
            "Compile Error": 5,
            "Presentation Error": 6,
            "Wa": 1,
            "Tle": 2,
            "Mle": 3,
            "Re": 4,
            "Ce": 5,
            "Pe": 6,
        }
        return mapping.get(normalized_verdict, -1)


if __name__ == "__main__":
    print("--- SubmissionDataset Demo (Option 1: Returning Dict for Predictor) ---")

    DATA_DIR = "temp_dataset_demo_data_v2"  # Use a new temp dir name
    os.makedirs(DATA_DIR, exist_ok=True)
    problem_csv_file = os.path.join(DATA_DIR, "final_problem_statements.csv")
    submissions_code_file = os.path.join(DATA_DIR, "submissions_DemoLang.csv")
    submission_stats_file = os.path.join(DATA_DIR, "submission_stats_DemoLang.csv")

    # Create Dummy CSV data
    problem_data = {
        "problem_id": ["p101", "p102"],
        "statement": ["Solve fizz.", "Sort numbers."],
        "input_spec": ["N", "Array"],
        "output_spec": ["Fizz", "Sorted"],
    }
    pd.DataFrame(problem_data).to_csv(problem_csv_file, index=False)
    submissions_code_data = {
        "problem_id": ["p101", "p102"],
        "submission_file": ["s1.py", "s2.py"],
        "code": ["print('fizz')", "sorted([3,1,2])"],
    }
    pd.DataFrame(submissions_code_data).to_csv(submissions_code_file, index=False)
    submission_stats_data = {
        "problem_id": ["p101", "p102"],
        "submission_id": ["s1", "s2"],
        "language": ["python", "python"],
        "verdict": ["Accepted", "Wrong Answer"],
        "runtime": [0.1, 0.2],
        "memory": [100, 120],
        "code_size": [10, 20],
    }
    pd.DataFrame(submission_stats_data).to_csv(submission_stats_file, index=False)

    print("\nInitializing Preprocessor and CodeNormalizer...")
    try:
        preprocessor = Preprocessor()
        code_normalizer = CodeNormalizer()
        print("Preprocessor and CodeNormalizer initialized.")

        print("\nInitializing SubmissionDataset...")
        dataset = SubmissionDataset(
            stats_csv_path=submission_stats_file,
            code_csv_path=submissions_code_file,
            problem_csv_path=problem_csv_file,
            preprocessor_instance=preprocessor,
            code_normalizer_instance=code_normalizer,
        )
        print(f"Dataset length: {len(dataset)}")

        if len(dataset) > 0:
            print("\nFetching first item from dataset:")
            item_dict = dataset[0]
            print(f"  Item type: {type(item_dict)}")
            print(f"  Keys: {list(item_dict.keys())}")
            print(f"  Code String (first 30 chars): '{item_dict['code_str'][:30]}...'")
            print(
                f"  Statement String (first 30 chars): '{item_dict['statement_str'][:30]}...'"
            )
            print(f"  Language: {item_dict['lang_str']}")
            print(f"  Verdict (Encoded): {item_dict['verdict_encoded'].item()}")
            print(f"  Verdict (Raw): {item_dict['verdict_raw_str']}")

    except ImportError as e_imp:
        print(f"CRITICAL IMPORT ERROR in demo: {e_imp}")
    except Exception as e:
        print(f"An error occurred during dataset demo: {e}")
        import traceback

        traceback.print_exc()
    # finally:
    #     if os.path.exists(DATA_DIR):
    #         import shutil
    #         shutil.rmtree(DATA_DIR)
    #         print(f"\nCleaned up dummy data directory: {DATA_DIR}")
