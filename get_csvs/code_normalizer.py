import re
from pathlib import Path

__all__ = ["CodeNormalizer"]

# File extensions by language for comment removal
_EXTS_LANG = {
    "C++": {".cpp", ".cc", ".cxx", ".C"},
    "Python": {".py"},
    "Java": {".java"},
}


class CodeNormalizer:
    """
    A utility class to normalize source code strings or files by removing comments
    and collapsing whitespace, for Python, C++, and Java.
    """

    @staticmethod
    def collapse_whitespace(s: str) -> str:
        """
        Replace any run of whitespace (spaces, tabs, newlines) with a single space.
        """
        return re.sub(r"\s+", " ", s).strip()

    @staticmethod
    def remove_comments(code: str, lang: str) -> str:
        """
        Strip comments from code according to language.

        - Python: '#' to end-of-line
        - C++/Java: '//' to end-of-line, and '/* ... */' block comments
        """
        if lang == "Python":
            # Remove '#' comments (naive, ignores strings)
            return re.sub(r"#.*", "", code)
        else:
            # Remove C-style line comments
            without_line = re.sub(r"//.*", "", code)
            # Remove block comments (non-greedy across lines)
            return re.sub(r"/\*.*?\*/", "", without_line, flags=re.DOTALL)

    @classmethod
    def normalize_code(cls, code: str, lang: str) -> str:
        """
        Perform full normalization on a code string.

        Steps:
        1. Remove comments appropriate to `lang`.
        2. Collapse all runs of whitespace to single spaces.
        3. Strip leading/trailing whitespace.

        Args:
            code: Raw code text.
            lang: Programming language (e.g., 'Python', 'C++', 'Java').

        Returns:
            A normalized code string.
        """
        stripped = cls.remove_comments(code, lang)
        cleaned = cls.collapse_whitespace(stripped)
        return cleaned

    @classmethod
    def normalize_file(self, filepath: str, lang: str) -> str:
        """Reads a file, normalizes its content using the given language, and returns the normalized code string."""
        try:
            with open(
                filepath, "r", encoding="utf-8", errors="ignore"
            ) as f:  # Add errors='ignore' for robustness
                code = f.read()
            # Now pass the lang argument to normalize_code
            return self.normalize_code(code, lang)
        except FileNotFoundError:
            # print(
            #     f"Warning (normalize_file): File not found at {filepath}. Returning empty string."
            # )
            return ""  # Return empty string for FileNotFoundError to be skipped in extract.py
        except Exception as e:
            # print(
            #     f"Warning (normalize_file): Error reading/normalizing file {filepath}: {e}. Returning empty string."
            # )
            return ""  # Return empty for other errors too
