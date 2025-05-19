import pandas as pd


def build_final_statements(metadata_csv, statements_csv, output_csv):
    # Load both CSVs
    meta_df = pd.read_csv(metadata_csv, usecols=["id"])
    stmt_df = pd.read_csv(
        statements_csv, usecols=["problem_id", "statement", "input_spec", "output_spec"]
    )

    # Compute intersection of IDs
    common_ids = set(meta_df["id"]).intersection(stmt_df["problem_id"])
    print(f"Found {len(common_ids)} problems present in both files.")

    # Filter statements to only those IDs
    final_df = stmt_df[stmt_df["problem_id"].isin(common_ids)].copy()

    # (Optional) sort by id
    final_df.sort_values("problem_id", inplace=True)

    # Write out
    final_df.to_csv(output_csv, index=False)
    print(f"✅ Wrote {len(final_df)} final statements to '{output_csv}'")


if __name__ == "__main__":
    build_final_statements(
        metadata_csv="filtered_problems_metadata.csv",
        statements_csv="problems_preprocessed.csv",
        output_csv="final_problem_statements.csv",
    )
