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
    def normalize_file(cls, filepath: str) -> str:
        """
        Normalize a source code file by path, auto-detecting language from extension.

        Args:
            filepath: Path to the source file.

        Returns:
            Normalized code string.
        """
        p = Path(filepath)
        ext = p.suffix
        # Detect language by extension
        lang = None
        for key, extset in _EXTS_LANG.items():
            if ext in extset:
                lang = key
                break
        if lang is None:
            raise ValueError(f"Unsupported file extension: {ext}")

        content = p.read_text(encoding="utf-8", errors="ignore")
        return cls.normalize_code(content, lang)
