import unittest

from fastapi.testclient import TestClient

from api import api


class TestDocumentQaApi(unittest.TestCase):
    def test_health_check(self):
        with TestClient(api) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "ok",
                "service": "document-qa",
            },
        )

    def test_ask_returns_answer_and_references(self):
        question = "如何 隔离 Python 项目 依赖"

        with TestClient(api) as client:
            response = client.post(
                "/ask",
                json={
                    "question": f"  {question}  ",
                },
            )

        self.assertEqual(response.status_code, 200)

        data = response.json()

        # 问题首尾空白应被清除
        self.assertEqual(data["question"], question)
        self.assertEqual(data["mode"], "mock")
        self.assertEqual(
            data["answer"],
            "这是模拟回答，尚未调用真实模型。",
        )

        # 未填写 top_k 时使用默认值 2
        self.assertEqual(len(data["references"]), 2)

        for reference in data["references"]:
            self.assertEqual(
                reference["source"],
                "python_venv.txt",
            )
            self.assertIn("path", reference)
            self.assertIn("text", reference)
            self.assertGreaterEqual(
                reference["score"],
                0.1,
            )

    def test_ask_respects_top_k(self):
        with TestClient(api) as client:
            response = client.post(
                "/ask",
                json={
                    "question": "Python",
                    "top_k": 1,
                    "min_score": 0.1,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            len(response.json()["references"]),
            1,
        )

    def test_rejects_invalid_request_fields(self):
        invalid_requests = [
            ("缺少问题", {}),
            ("空问题", {"question": ""}),
            ("空白问题", {"question": " \n "}),
            ("问题过长", {"question": "a" * 2001}),
            (
                "top_k 小于下限",
                {"question": "Python", "top_k": 0},
            ),
            (
                "top_k 大于上限",
                {"question": "Python", "top_k": 11},
            ),
            (
                "top_k 类型错误",
                {"question": "Python", "top_k": "abc"},
            ),
            (
                "min_score 小于下限",
                {"question": "Python", "min_score": -0.1},
            ),
            (
                "min_score 大于上限",
                {"question": "Python", "min_score": 1.1},
            ),
        ]

        with TestClient(api) as client:
            for description, payload in invalid_requests:
                with self.subTest(case=description):
                    response = client.post(
                        "/ask",
                        json=payload,
                    )

                    self.assertEqual(
                        response.status_code,
                        422,
                    )
                    self.assertIn(
                        "detail",
                        response.json(),
                    )

    def test_returns_error_for_unknown_question(self):
        with TestClient(api) as client:
            response = client.post(
                "/ask",
                json={
                    "question": "astronomy",
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["detail"],
            "question does not contain indexed terms",
        )

    def test_returns_error_when_threshold_filters_all_chunks(self):
        with TestClient(api) as client:
            response = client.post(
                "/ask",
                json={
                    "question": "Python",
                    "min_score": 1.0,
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", response.json())


if __name__ == "__main__":
    unittest.main()
