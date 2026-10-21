"""Ensure public project code stays documented."""

import ast
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DocstringTests(unittest.TestCase):
    """Check every Python module, class, and function for documentation."""

    def test_python_modules_and_definitions_have_docstrings(self):
        """Require docstrings across application, script, and test sources."""
        missing = []
        for path in PROJECT_ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if ast.get_docstring(tree) is None:
                missing.append(f"{path.relative_to(PROJECT_ROOT)}: module")
            for node in ast.walk(tree):
                if (
                    isinstance(
                        node,
                        (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
                    )
                    and ast.get_docstring(node) is None
                ):
                    missing.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: {node.name}"
                    )

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
