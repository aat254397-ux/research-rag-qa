import unittest

from text_chunker import (
    chunk_document,
    validate_chunk_settings,
)


class TestTextChunker(unittest.TestCase):
    def setUp(self):
        self.loaded_document = {
            "source": "example.txt",
            "path": "sample_docs/example.txt",
            "text": "ABCDEFGHIJ",
        }

    def test_creates_overlapping_chunks(self):
        document_chunks = chunk_document(
            self.loaded_document,
            chunk_size=4,
            overlap=1,
        )

        self.assertEqual(
            [
                document_chunk["text"]
                for document_chunk in document_chunks
            ],
            [
                "ABCD",
                "DEFG",
                "GHIJ",
            ],
        )

    def test_records_chunk_positions(self):
        document_chunks = chunk_document(
            self.loaded_document,
            chunk_size=4,
            overlap=1,
        )

        self.assertEqual(
            [
                (
                    document_chunk["start"],
                    document_chunk["end"],
                )
                for document_chunk in document_chunks
            ],
            [
                (0, 4),
                (3, 7),
                (6, 10),
            ],
        )

    def test_preserves_source_metadata(self):
        document_chunks = chunk_document(
            self.loaded_document,
            chunk_size=4,
            overlap=1,
        )

        self.assertEqual(
            document_chunks[0]["chunk_id"],
            1,
        )
        self.assertEqual(
            document_chunks[0]["source"],
            "example.txt",
        )
        self.assertEqual(
            document_chunks[0]["path"],
            "sample_docs/example.txt",
        )

    def test_rejects_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            validate_chunk_settings(
                chunk_size=0,
                overlap=0,
            )

    def test_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            validate_chunk_settings(
                chunk_size=4,
                overlap=4,
            )


if __name__ == "__main__":
    unittest.main()
