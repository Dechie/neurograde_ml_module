#!/usr/bin/env python3
import pandas as pd
import os
import json


def csv_to_json_folder(csv_path: str, output_folder: str):
    """
    Reads a CSV file and writes each row as a separate JSON file named <submission_id>.json
    into the specified output_folder.
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate rows
    for _, row in df.iterrows():
        submission_id = row["submission_id"]
        record = row.to_dict()
        out_path = os.path.join(output_folder, f"{submission_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
    print(
        f"Finished exporting {len(df)} records from '{csv_path}' to '{output_folder}/'"
    )


def main():
    # Map your CSV filenames to the desired output folders
    csv_files = {
        "data/analysis_correct_python.csv": "../correct_jsons",
        "data/analysis_incorrect_python.csv": "../incorrect_jsons",
    }

    for csv_path, out_folder in csv_files.items():
        csv_to_json_folder(csv_path, out_folder)


if __name__ == "__main__":
    main()
