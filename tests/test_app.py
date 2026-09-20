import unittest

from app import (
    build_document_qa_index,
    build_document_qa_prompt,
)


class TestDocumentQaApplication(unittest.TestCase):
    def setUp(self):
        self.document_paths = [
            "sample_docs/python_venv.txt",
            "sample_docs/rag_notes.md",
        ]

    def test_builds_shared_index_from_multiple_documents(self):
        retrieval_index = build_document_qa_index(
            self.document_paths,
            chunk_size=80,
            overlap=20,
        )

        indexed_sources = {
            indexed_chunk["source"]
            for indexed_chunk in retrieval_index["chunks"]
        }

        self.assertEqual(
            len(retrieval_index["chunks"]),
            5,
        )
        self.assertEqual(
            indexed_sources,
            {
                "python_venv.txt",
                "rag_notes.md",
            },
        )

    def test_builds_prompt_from_question_and_index(self):
        retrieval_index = build_document_qa_index(
            self.document_paths,
            chunk_size=80,
            overlap=20,
        )
        question_text = (
            "如何 隔离 Python 项目 依赖"
        )

        prompt = build_document_qa_prompt(
            question_text,
            retrieval_index,
            top_k=2,
            min_score=0.1,
        )

        self.assertIn(
            "参考资料：",
            prompt,
        )
        self.assertIn(
            "来源：python_venv.txt",
            prompt,
        )
        self.assertIn(
            question_text,
            prompt,
        )
        self.assertTrue(
            prompt.endswith("回答：")
        )

    def test_rejects_empty_document_paths(self):
        with self.assertRaises(ValueError):
            build_document_qa_index([])


if __name__ == "__main__":
    unittest.main()
