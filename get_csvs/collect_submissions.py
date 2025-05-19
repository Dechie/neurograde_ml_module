import csv
import re
import pandas as pd
from pathlib import Path


def collapse_whitespace(s: str) -> str:
    """Replace any run of whitespace (spaces, tabs, newlines) with a single space."""
    return re.sub(r"\s+", " ", s).strip()


def remove_comments(code: str, lang: str) -> str:
    """
    Strip out comments from code according to language.
    - Python: '#' to end‑of‑line
    - C++/Java: '//' to end‑of‑line, and '/* ... */' block comments
    """
    if lang == "Python":
        # Remove '#' comments but leave '#' in strings untouched in the simple case
        return re.sub(r"#.*", "", code)
    else:
        # Remove C‑style line comments
        code_no_line = re.sub(r"//.*", "", code)
        # Remove C‑style block comments (non‑greedy, across lines)
        return re.sub(r"/\*.*?\*/", "", code_no_line, flags=re.DOTALL)


def collect_submissions(final_ids_csv: str, lang_dirs: dict, output_dir: str = "."):
    # Load the list of final problem IDs
    final_ids = set(pd.read_csv(final_ids_csv)["problem_id"].astype(str))
    print(f"🔍 Loaded {len(final_ids)} final problem IDs.")

    # File extensions by language
    exts_map = {
        "C++": {".cpp", ".cc", ".cxx", ".C"},
        "Python": {".py"},
        "Java": {".java"},
    }

    for lang, folder in lang_dirs.items():
        records = []
        lang_path = Path(folder)
        if not lang_path.exists():
            print(f"⚠️  Directory for {lang} not found at {folder}, skipping.")
            continue

        print(f"📂 Processing {lang} submissions in {folder}...")

        # Each subfolder named by problem_id
        for prob_dir in lang_path.iterdir():
            pid = prob_dir.name
            if pid not in final_ids or not prob_dir.is_dir():
                continue

            for file in prob_dir.iterdir():
                if file.is_file() and file.suffix in exts_map.get(lang, ()):
                    try:
                        raw = file.read_text(encoding="utf-8", errors="ignore")
                    except Exception as e:
                        print(f"   ⚠️  Could not read {file}: {e}")
                        continue

                    # 1) strip comments, 2) collapse whitespace
                    stripped = remove_comments(raw, lang)
                    code_clean = collapse_whitespace(stripped)

                    records.append(
                        {
                            "problem_id": pid,
                            "submission_file": file.name,
                            "code": code_clean,
                        }
                    )

        # Write out CSV
        out_name = f"submissions_{lang.replace('+','p')}.csv"
        out_path = Path(output_dir) / out_name
        df = pd.DataFrame.from_records(
            records, columns=["problem_id", "submission_file", "code"]
        )
        df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"✅ Wrote {len(df)} {lang} submissions to {out_path}")


if __name__ == "__main__":
    lang_dirs = {
        "C++": "../cpp_subs",
        "Python": "../python_subs",
        "Java": "../java_subs",
    }

    collect_submissions(
        final_ids_csv="final_problem_statements.csv",
        lang_dirs=lang_dirs,
        output_dir=".",  # or "outputs/"
    )
