import os
import pandas as pd


def collect_problem_ids(root_dirs):
    """
    Traverse each directory in root_dirs and collect folder names
    matching 'p' followed by digits (e.g., p00001).
    """
    ids = set()
    for root in root_dirs:
        for dirpath, dirnames, _ in os.walk(root):
            for dirname in dirnames:
                if dirname.startswith("p") and dirname[1:].isdigit():
                    ids.add(dirname)
    return sorted(ids)


def main():
    # 1. Define benchmark folder paths
    roots = ["../cpp_subs", "../python_subs", "../java_subs"]

    # 2. Collect problem IDs
    problem_ids = collect_problem_ids(roots)
    print(f"Found {len(problem_ids)} problems across benchmarks.")

    # 3. Load global metadata
    metadata_df = pd.read_csv("../Project_CodeNet/metadata/problem_list.csv")

    # 4. Filter metadata to only selected problems
    filtered_df = metadata_df[metadata_df["id"].isin(problem_ids)]
    del filtered_df['rating']
    del filtered_df['tags']
    del filtered_df['complexity']
    print(f"Filtered metadata contains {len(filtered_df)} entries.")

    # 5. Export to new CSV
    output_path = "filtered_problems_metadata.csv"
    filtered_df.to_csv(output_path, index=False)
    print(f"Exported filtered metadata to {output_path}")


if __name__ == "__main__":
    main()
