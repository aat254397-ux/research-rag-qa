import unittest

from retrieval import (
    build_retrieval_index,
    build_vocabulary,
    document_frequency,
    inverse_document_frequency,
    text_to_tfidf_vector,
    tokenize_text,
)


class TestRetrievalIndex(unittest.TestCase):
    def setUp(self):
        self.knowledge_chunks = [
            {
                "chunk_id": 1,
                "source": "a.txt",
                "path": "sample_docs/a.txt",
                "text": "Python Python 项目",
                "start": 0,
                "end": 16,
            },
            {
                "chunk_id": 2,
                "source": "a.txt",
                "path": "sample_docs/a.txt",
                "text": "Python 虚拟环境",
                "start": 12,
                "end": 25,
            },
            {
                "chunk_id": 1,
                "source": "b.txt",
                "path": "sample_docs/b.txt",
                "text": "Git 项目",
                "start": 0,
                "end": 6,
            },
        ]

    def test_tokenizes_lowercase_and_punctuation(self):
        words = tokenize_text(
            "Python，项目！ Top-K."
        )

        self.assertEqual(
            words,
            [
                "python",
                "项目",
                "top",
                "k",
            ],
        )

    def test_builds_unique_sorted_vocabulary(self):
        vocabulary = build_vocabulary(
            self.knowledge_chunks
        )

        self.assertEqual(
            vocabulary,
            sorted(set(vocabulary)),
        )
        self.assertEqual(
            set(vocabulary),
            {
                "python",
                "项目",
                "虚拟环境",
                "git",
            },
        )

    def test_document_frequency_counts_chunks(self):
        python_frequency = document_frequency(
            "python",
            self.knowledge_chunks,
        )

        self.assertEqual(
            python_frequency,
            2,
        )

    def test_rare_term_has_higher_idf(self):
        common_term_weight = (
            inverse_document_frequency(
                "python",
                self.knowledge_chunks,
            )
        )
        rare_term_weight = (
            inverse_document_frequency(
                "虚拟环境",
                self.knowledge_chunks,
            )
        )

        self.assertGreater(
            rare_term_weight,
            common_term_weight,
        )

    def test_creates_tfidf_vector(self):
        vocabulary = [
            "python",
            "项目",
            "虚拟环境",
        ]
        idf_weights = {
            "python": 1.0,
            "项目": 2.0,
            "虚拟环境": 3.0,
        }

        text_vector = text_to_tfidf_vector(
            "Python Python 项目",
            vocabulary,
            idf_weights,
        )

        self.assertEqual(
            text_vector,
            [
                2.0,
                2.0,
                0.0,
            ],
        )

    def test_builds_index_and_preserves_metadata(self):
        retrieval_index = build_retrieval_index(
            self.knowledge_chunks
        )

        indexed_chunk = retrieval_index["chunks"][0]

        self.assertEqual(
            indexed_chunk["source"],
            "a.txt",
        )
        self.assertEqual(
            len(indexed_chunk["vector"]),
            len(retrieval_index["vocabulary"]),
        )
        self.assertIn(
            "vector",
            indexed_chunk,
        )
        self.assertNotIn(
            "vector",
            self.knowledge_chunks[0],
        )

    def test_rejects_empty_knowledge_chunks(self):
        with self.assertRaises(ValueError):
            build_retrieval_index([])


if __name__ == "__main__":
    unittest.main()
