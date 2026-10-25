import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from api import create_app


class TestIndexPersistenceApi(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.snapshot_path = self.directory / "index.json"
        self.document_path = self.directory / "paper_notes.txt"
        self.document_path.write_text(
            "Python projects use virtual environments. 中文资料。",
            encoding="utf-8",
        )

    def test_new_application_restores_saved_index_without_source_file(self):
        first_application = create_app(index_path=self.snapshot_path)
        with TestClient(first_application) as client:
            initial_status = client.get("/index/status").json()
            self.assertEqual(initial_status["index_origin"], "sample_documents")
            self.assertFalse(initial_status["snapshot_exists"])

            response = client.post(
                "/documents/load",
                json={"paths": [str(self.document_path)]},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["index_storage"], "memory+disk")
            previous_status = client.get("/index/status").json()
            response = client.post(
                "/ask", json={"question": "Python", "mode": "mock"}
            )
            self.assertEqual(response.status_code, 200)
            first_answer = response.json()

        # 只删除本测试创建的临时资料，验证恢复时确实不重读原文。
        self.document_path.unlink()

        second_application = create_app(index_path=self.snapshot_path)
        with TestClient(second_application) as client:
            response = client.get("/index/status")
            self.assertEqual(response.status_code, 200)
            restored_status = response.json()
            self.assertEqual(restored_status["index_origin"], "disk")
            self.assertTrue(restored_status["snapshot_exists"])
            self.assertGreater(restored_status["snapshot_size_bytes"], 0)
            self.assertEqual(
                restored_status["documents"], previous_status["documents"]
            )
            self.assertEqual(
                restored_status["chunk_count"], previous_status["chunk_count"]
            )

            response = client.post(
                "/ask", json={"question": "Python", "mode": "mock"}
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json()["references"], first_answer["references"]
            )
            self.assertEqual(
                response.json()["references"][0]["source"], "paper_notes.txt"
            )

    def test_failed_document_load_keeps_current_index_and_snapshot(self):
        application = create_app(index_path=self.snapshot_path)
        with TestClient(application) as client:
            response = client.post(
                "/documents/load",
                json={"paths": [str(self.document_path)]},
            )
            self.assertEqual(response.status_code, 200)
            previous_status = client.get("/index/status").json()
            previous_bytes = self.snapshot_path.read_bytes()

            response = client.post(
                "/documents/load",
                json={"paths": [str(self.directory / "missing.pdf")]},
            )

            self.assertEqual(response.status_code, 404)
            self.assertEqual(client.get("/index/status").json(), previous_status)
            self.assertEqual(self.snapshot_path.read_bytes(), previous_bytes)

    def test_invalid_snapshot_stops_startup_instead_of_using_sample_library(self):
        self.snapshot_path.write_text("{", encoding="utf-8")
        application = create_app(index_path=self.snapshot_path)

        with self.assertRaisesRegex(RuntimeError, "无法恢复索引快照"):
            with TestClient(application):
                pass

        self.assertEqual(self.snapshot_path.read_text(encoding="utf-8"), "{")


if __name__ == "__main__":
    unittest.main()
