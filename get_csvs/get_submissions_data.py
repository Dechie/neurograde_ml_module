import pandas as pd
import csv
from pathlib import Path


def load_submission_map(sub_csv: str):
    """
    Reads a submissions CSV with 'problem_id' and 'submission_file' columns.
    Returns a dict mapping problem_id -> set of submission_id strings
    (filename stem, i.e. without extension).
    """
    df = pd.read_csv(sub_csv, dtype=str, usecols=["problem_id", "submission_file"])
    df["submission_id"] = df["submission_file"].apply(lambda fn: Path(fn).stem)
    mapping = df.groupby("problem_id")["submission_id"].apply(set).to_dict()
    return mapping


def filter_metadata_by_submissions(
    final_ids_csv: str,
    metadata_dir: str,
    submissions_csv_map: dict,
    output_dir: str = ".",
):
    """
    For each language in submissions_csv_map, load its submission map,
    then walk through metadata/<pXXXXX>.csv, keep only rows where
    submission_id in that map, and write out a filtered CSV.
    """
    # Load the final set of problems
    final_ids = set(
        pd.read_csv(final_ids_csv, usecols=["problem_id"])["problem_id"].astype(str)
    )
    print(f"🔍 Tracking metadata for {len(final_ids)} final problems.")

    for lang, sub_csv in submissions_csv_map.items():
        sub_map = load_submission_map(sub_csv)
        records = []

        for meta_file in Path(metadata_dir).glob("p*.csv"):
            pid = meta_file.stem
            if pid not in final_ids or pid not in sub_map:
                continue

            df = pd.read_csv(meta_file, dtype=str)
            df.columns = [c.lower() for c in df.columns]
            df["submission_id"] = df["submission_id"].astype(str)

            keep_ids = sub_map[pid]
            df_filt = df[df["submission_id"].isin(keep_ids)]
            if df_filt.empty:
                continue

            print(df_filt["status"].value_counts(dropna=True))

            rec = df_filt.loc[
                :,
                [
                    "submission_id",
                    "status",
                    "cpu_time",
                    "memory",
                    "code_size",
                ],
            ].rename(columns={"status": "verdict", "cpu_time": "runtime"})
            rec.insert(0, "problem_id", pid)
            rec.insert(2, "language", lang)
            records.append(rec)

        if not records:
            print(f"⚠️  No metadata matched for {lang}")
            continue

        # concatenate and post-process
        out_df = pd.concat(records, ignore_index=True)
        # coerce numeric fields
        out_df["runtime"] = pd.to_numeric(out_df["runtime"], errors="coerce")
        out_df["memory"] = pd.to_numeric(out_df["memory"], errors="coerce")

        # write out
        out_name = f"submission_stats_{lang.replace('+','p')}.csv"
        out_path = Path(output_dir) / out_name
        out_df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"✅ Wrote {len(out_df)} records for {lang} → {out_path}")


if __name__ == "__main__":
    submissions_csv_map = {
        "C++": "data/submissions_Cpp.csv",
        "Python": "data/submissions_Python.csv",
        "Java": "data/submissions_Java.csv",
    }
    filter_metadata_by_submissions(
        final_ids_csv="./data/final_problem_statements.csv",
        metadata_dir="../Project_CodeNet/metadata",
        submissions_csv_map=submissions_csv_map,
        output_dir=".",
    )
