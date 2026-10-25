import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from index_store import (
    FORMAT_VERSION,
    INDEX_TYPE,
    load_retrieval_index,
    save_retrieval_index,
)
from retrieval import build_retrieval_index, retrieve_relevant_chunks


class TestIndexStore(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.snapshot_path = (
            Path(self.temporary_directory.name) / "index.json"
        )
        text = "MJO 指数 uses OLR and zonal wind."
        self.index = build_retrieval_index([
            {
                "source": "论文.pdf",
                "path": "papers/论文.pdf",
                "page": 3,
                "chunk_id": 1,
                "start": 0,
                "end": len(text),
                "text": text,
            },
        ])

    def test_round_trip_preserves_index_and_retrieval_results(self):
        save_retrieval_index(self.index, self.snapshot_path)
        restored_index = load_retrieval_index(self.snapshot_path)

        self.assertEqual(restored_index, self.index)
        self.assertEqual(restored_index["chunks"][0]["page"], 3)
        self.assertEqual(
            retrieve_relevant_chunks("MJO OLR", restored_index),
            retrieve_relevant_chunks("MJO OLR", self.index),
        )
        self.assertGreater(self.snapshot_path.stat().st_size, 0)

    def test_rejects_incompatible_snapshot_version(self):
        payload = {
            "format_version": FORMAT_VERSION + 1,
            "index_type": INDEX_TYPE,
            "index": self.index,
        }
        self.snapshot_path.write_text(
            json.dumps(payload), encoding="utf-8"
        )

        with self.assertRaisesRegex(ValueError, "unsupported"):
            load_retrieval_index(self.snapshot_path)

    def test_rejects_snapshot_with_wrong_vector_length(self):
        damaged_index = deepcopy(self.index)
        damaged_index["chunks"][0]["vector"].pop()
        payload = {
            "format_version": FORMAT_VERSION,
            "index_type": INDEX_TYPE,
            "index": damaged_index,
        }
        self.snapshot_path.write_text(
            json.dumps(payload), encoding="utf-8"
        )

        with self.assertRaisesRegex(ValueError, "vector"):
            load_retrieval_index(self.snapshot_path)

    def test_failed_write_preserves_previous_snapshot(self):
        save_retrieval_index(self.index, self.snapshot_path)
        previous_bytes = self.snapshot_path.read_bytes()

        # object 无法写成 JSON：让写入中途失败，不接触真实论文文件。
        invalid_index = deepcopy(self.index)
        invalid_index["unsupported_extra_field"] = object()
        with self.assertRaises(TypeError):
            save_retrieval_index(invalid_index, self.snapshot_path)

        self.assertEqual(self.snapshot_path.read_bytes(), previous_bytes)
        self.assertEqual(load_retrieval_index(self.snapshot_path), self.index)
        self.assertEqual(
            list(self.snapshot_path.parent.iterdir()),
            [self.snapshot_path],
        )


if __name__ == "__main__":
    unittest.main()
