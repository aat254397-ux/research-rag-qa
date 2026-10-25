import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from api import create_app
from app import build_document_qa_index
from embedding_retrieval import DIMENSION, ENCODER_SPEC, corpus_fingerprint
from embedding_retrieval import retrieve_embedding_chunks
from embedding_store import (
    load_embedding_index,
    save_embedding_index,
    validate_embedding_index,
)
from index_store import save_retrieval_index


class TestEmbeddingIndex(unittest.TestCase):
    def setUp(self):
        # 只验证向量数学、存储格式和资料版本保护，不模拟模型语义能力。
        self.chunks = []
        for chunk_id, text in enumerate(("MJO OLR wind", "Python environment"), start=1):
            self.chunks.append({
                "source": "example.pdf",
                "path": "papers/example.pdf",
                "page": chunk_id,
                "chunk_id": chunk_id,
                "start": 0,
                "end": len(text),
                "text": text,
            })
        self.query_vector = [1.0] + [0.0] * (DIMENSION - 1)
        self.index = {
            "encoder_spec": dict(ENCODER_SPEC),
            "source_fingerprint": corpus_fingerprint(self.chunks),
            "chunks": [
                {**self.chunks[0], "vector": self.query_vector.copy()},
                {**self.chunks[1], "vector": [-1.0] + [0.0] * (DIMENSION - 1)},
            ],
        }

    def test_ranks_signed_vectors_without_mutating_index(self):
        original_index = deepcopy(self.index)
        results = retrieve_embedding_chunks(
            self.query_vector, self.index, top_k=2, min_score=-1.0
        )

        self.assertEqual([result["chunk_id"] for result in results], [1, 2])
        self.assertAlmostEqual(results[0]["score"], 1.0)
        self.assertAlmostEqual(results[1]["score"], -1.0)
        self.assertEqual(results[0]["page"], 1)
        self.assertEqual(self.index, original_index)

    def test_embedding_snapshot_preserves_metadata_and_signed_vectors(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "index.e5.json"
            save_embedding_index(self.index, path)
            restored_index = load_embedding_index(path)

        self.assertEqual(restored_index, self.index)

    def test_rejects_different_encoder_configuration(self):
        changed_index = deepcopy(self.index)
        changed_index["encoder_spec"]["revision"] = "another-model-version"

        with self.assertRaisesRegex(ValueError, "configuration"):
            validate_embedding_index(changed_index)

    def test_rejects_wrong_vector_dimension(self):
        changed_index = deepcopy(self.index)
        changed_index["chunks"][0]["vector"].pop()

        with self.assertRaisesRegex(ValueError, "dimension"):
            validate_embedding_index(changed_index)

    def test_changed_text_invalidates_source_fingerprint(self):
        changed_index = deepcopy(self.index)
        changed_index["chunks"][0]["text"] = "Updated MJO definition"

        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_embedding_index(changed_index)


class TestEmbeddingApiState(unittest.TestCase):
    def test_unbuilt_embedding_index_is_not_silently_replaced_by_tfidf(self):
        with TestClient(create_app(index_path=None)) as client:
            response = client.post(
                "/search",
                json={"question": "Python", "retriever": "embedding"},
            )
            self.assertEqual(response.status_code, 409)
            self.assertIn("/index/embedding/build", response.json()["detail"])
            self.assertFalse(client.get("/index/status").json()["embedding"]["model_loaded"])

    def test_changed_library_blocks_stale_embedding_snapshot(self):
        with TemporaryDirectory() as directory:
            directory_path = Path(directory)
            first_document = directory_path / "first.txt"
            first_document.write_text("MJO OLR wind", encoding="utf-8")
            second_document = directory_path / "second.txt"
            second_document.write_text("Python environment", encoding="utf-8")
            snapshot_path = directory_path / "index.json"
            embedding_path = directory_path / "index.e5.json"
            tfidf_index = build_document_qa_index([first_document])
            save_retrieval_index(tfidf_index, snapshot_path)

            # 这是存储协议的单位向量样本，不调用 E5，也不假装验证检索语义。
            chunk = {key: value for key, value in tfidf_index["chunks"][0].items() if key != "vector"}
            embedding_index = {
                "encoder_spec": dict(ENCODER_SPEC),
                "source_fingerprint": corpus_fingerprint([chunk]),
                "chunks": [{**chunk, "vector": [1.0] + [0.0] * (DIMENSION - 1)}],
            }
            save_embedding_index(embedding_index, embedding_path)
            previous_embedding_bytes = embedding_path.read_bytes()

            with TestClient(create_app(index_path=snapshot_path)) as client:
                status = client.get("/index/status").json()["embedding"]
                self.assertEqual(status["status"], "ready")
                self.assertEqual(status["origin"], "disk")
                self.assertFalse(status["model_loaded"])

                response = client.post(
                    "/documents/load", json={"paths": [str(second_document)]}
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    client.get("/index/status").json()["embedding"]["status"], "stale"
                )
                response = client.post(
                    "/search",
                    json={"question": "Python", "retriever": "embedding", "min_score": -1},
                )
                self.assertEqual(response.status_code, 409)

            self.assertEqual(embedding_path.read_bytes(), previous_embedding_bytes)

            # 重启同样识别旧快照，不需要加载任何模型权重。
            with TestClient(create_app(index_path=snapshot_path)) as client:
                self.assertEqual(
                    client.get("/index/status").json()["embedding"]["status"], "stale"
                )


if __name__ == "__main__":
    unittest.main()
