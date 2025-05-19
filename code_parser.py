import sys
from typing import Dict, Any, Optional, List

from tree_sitter import Tree, Node
from tree_sitter_language_pack import get_parser

SUPPORTED_LANGUAGES = {
    "python": "python",
    "java": "java",
    "cpp": "cpp",
    "c++": "cpp",
}


class CodeParser:
    def __init__(self):
        self.parsers: Dict[str, Any] = {}
        for key, lang_name in SUPPORTED_LANGUAGES.items():
            try:
                self.parsers[key] = get_parser(lang_name)
            except Exception as e:
                print(f"Warning: Could not load parser for '{key}': {e}")

        if not self.parsers:
            print(
                "Warning: No parsers loaded. "
                "Ensure tree-sitter and tree-sitter-language-pack are installed."
            )

    def get_supported_languages(self) -> List[str]:
        return list(self.parsers.keys())

    def parse(self, code: str, lang: str) -> Optional[Tree]:
        if not isinstance(code, str):
            raise TypeError("Input 'code' must be a string.")
        if not isinstance(lang, str):
            raise TypeError("Input 'lang' must be a string.")

        normalized = lang.strip().lower()
        if not normalized:
            raise ValueError("'lang' argument cannot be empty.")

        parser = self.parsers.get(normalized)
        if not parser:
            avail = ", ".join(self.get_supported_languages()) or "None"
            raise ValueError(f"Unsupported language: '{lang}'. Available: [{avail}].")

        try:
            return parser.parse(code.encode("utf8"))
        except Exception as e:
            print(f"Error parsing '{normalized}': {e}")
            return None

    def _node_to_dict(self, node: Node, code_bytes: bytes) -> Dict[str, Any]:
        return {
            "type": node.type,
            "text": code_bytes[node.start_byte : node.end_byte].decode(
                "utf8", errors="replace"
            ),
            "start_byte": node.start_byte,
            "end_byte": node.end_byte,
            "start_point": node.start_point,
            "end_point": node.end_point,
            "is_named": node.is_named,
            "is_error": node.is_error or node.type == "ERROR" or node.is_missing,
            "is_missing": node.is_missing,
            "children": [
                self._node_to_dict(child, code_bytes) for child in node.children
            ],
        }

    def get_ast_dict(self, code: str, lang: str) -> Optional[Dict[str, Any]]:
        tree = self.parse(code, lang)
        if not tree:
            return None
        root = tree.root_node
        ast = self._node_to_dict(root, code.encode("utf8"))
        ast["has_syntax_errors"] = root.has_error
        return ast


if __name__ == "__main__":
    print("--- CodeParser Demo (language‑pack) ---")
    parser = CodeParser()
    print("Supported languages:", parser.get_supported_languages())

    samples = {
        "python": "def hello():\n    print('Hello!')\n",
        "java": 'public class Test { public static void main(String[] a){ System.out.println("Hi"); } }',
        "cpp": '#include <iostream>\nint main(){ std::cout<<"C++!"; }',
    }

    for lang_key, code in samples.items():
        if lang_key not in parser.get_supported_languages():
            print(f"Skip {lang_key}: parser not loaded.")
            continue
        print(f"\n--- Testing {lang_key} ---")
        ast = parser.get_ast_dict(code, lang_key)
        if ast:
            print(f"Root type: {ast['type']}, Errors: {ast['has_syntax_errors']}")
        else:
            print("Failed to generate AST.")
