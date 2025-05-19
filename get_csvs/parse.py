#!/usr/bin/env python3
import csv
from tree_sitter_language_pack import get_parser


def read_first_code(csv_path: str) -> str:
    """Read the first non-header row's 'code' cell from a CSV."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
        if not row:
            raise ValueError(f"No data in {csv_path}")
        return row["code"]


def print_code_snippet(code: str, lang_name: str):
    """Print the raw code snippet for inspection."""
    separator = "=" * 10
    print(f"\n{separator} {lang_name} Code Snippet {separator}\n")
    print(code)
    print(f"\n{separator}{separator}{separator}{separator}{separator}\n")


def print_ast(code: str, parser, lang_name: str):
    """Parse 'code' and print its AST nodes recursively."""
    tree = parser.parse(code.encode("utf-8"))
    root = tree.root_node

    def recurse(node, indent=0):
        prefix = "  " * indent
        print(f"{prefix}{node.type} [{node.start_point}–{node.end_point}]")
        for child in node.children:
            recurse(child, indent + 1)

    print(f"\n=== {lang_name} AST ===")
    recurse(root)


def main():
    # Prepare parsers for each language
    parsers = {
        "Python": get_parser("python"),
        "Java": get_parser("java"),
        "C++": get_parser("cpp"),
    }

    # CSV filenames
    files = {
        "Python": "submissions_Python.csv",
        "Java": "submissions_Java.csv",
        "C++": "submissions_Cpp.csv",
    }

    # Read, print code, then parse & print AST for each
    for lang, path in files.items():
        print(f"\n--- Processing {lang} from {path} ---")
        code_snip = read_first_code(path)
        print_code_snippet(code_snip, lang)
        print_ast(code_snip, parsers[lang], lang)


if __name__ == "__main__":
    main()
