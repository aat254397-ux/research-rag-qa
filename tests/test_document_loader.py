import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from document_loader import load_document


class TestDocumentLoader(unittest.TestCase):
    def test_loads_txt_document(self):
        loaded_document = load_document(
            "sample_docs/python_venv.txt"
        )

        self.assertEqual(
            loaded_document["source"],
            "python_venv.txt",
        )

        self.assertIn(
            "Python 虚拟环境",
            loaded_document["text"],
        )

    def test_loads_markdown_document(self):
        loaded_document = load_document(
            "sample_docs/rag_notes.md"
        )

        self.assertEqual(
            loaded_document["source"],
            "rag_notes.md",
        )

        self.assertIn(
            "RAG",
            loaded_document["text"],
        )

    def test_rejects_missing_document(self):
        with self.assertRaises(FileNotFoundError):
            load_document(
                "sample_docs/missing.txt"
            )

    def test_rejects_directory_path(self):
        with self.assertRaises(ValueError):
            load_document(
                "sample_docs"
            )

    def test_rejects_unsupported_extension(self):
        with self.assertRaises(ValueError):
            load_document(
                "document_loader.py"
            )

    def test_rejects_empty_document(self):
        with TemporaryDirectory() as temporary_directory:
            empty_document_path = (
                Path(temporary_directory) / "empty.txt"
            )

            empty_document_path.write_text(
                "   \n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_document(
                    empty_document_path
                )


if __name__ == "__main__":
    unittest.main()
