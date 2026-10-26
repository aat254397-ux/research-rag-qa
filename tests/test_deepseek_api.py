import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from httpx2 import Request, Response
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError

from api import create_app


api = create_app(index_path=None)


class TestDeepSeekApi(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "question": "如何 隔离 Python 项目 依赖",
            "top_k": 2,
            "min_score": 0.1,
            "mode": "deepseek",
        }

        self.fake_client = Mock()
        self.fake_client.responses.create.return_value = (
            SimpleNamespace(
                output_text="  这是测试生成的答案。\n"
            )
        )

        # 临时替换客户端创建函数
        self.client_patch = patch(
            "api.create_deepseek_client",
            return_value=(self.fake_client, "test-model"),
        )
        self.create_client = self.client_patch.start()

        # 每个测试结束后恢复原函数
        self.addCleanup(self.client_patch.stop)

        # 只创建请求信息，供模拟异常使用，不发送请求
        self.upstream_request = Request(
            "POST",
            "https://api.deepseek.com/responses",
        )

    def test_returns_answer_and_closes_client(self):
        with TestClient(api) as client:
            response = client.post(
                "/ask",
                json=self.payload,
            )

        self.assertEqual(response.status_code, 200)

        data = response.json()

        self.assertEqual(data["mode"], "deepseek")
        self.assertEqual(
            data["answer"],
            "这是测试生成的答案。",
        )
        self.assertEqual(len(data["references"]), 2)

        self.create_client.assert_called_once_with()
        self.fake_client.responses.create.assert_called_once()
        self.fake_client.close.assert_called_once_with()

    def test_returns_503_for_missing_configuration(self):
        self.create_client.side_effect = ValueError(
            "DEEPSEEK_API_KEY must not be empty"
        )

        with TestClient(api) as client:
            response = client.post(
                "/ask",
                json=self.payload,
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("配置不完整", response.json()["detail"])

        self.fake_client.responses.create.assert_not_called()
        self.fake_client.close.assert_not_called()

    def test_returns_504_for_timeout(self):
        self.fake_client.responses.create.side_effect = (
            APITimeoutError(
                request=self.upstream_request,
            )
        )

        with TestClient(api) as client:
            response = client.post(
                "/ask",
                json=self.payload,
            )

        self.assertEqual(response.status_code, 504)
        self.assertIn("超时", response.json()["detail"])
        self.fake_client.close.assert_called_once_with()

    def test_returns_502_for_model_service_error(self):
        self.fake_client.responses.create.side_effect = APIError(
            "模拟的上游错误详情",
            request=self.upstream_request,
            body=None,
        )

        with TestClient(api) as client:
            response = client.post(
                "/ask",
                json=self.payload,
            )

        self.assertEqual(response.status_code, 502)
        self.assertIn("调用失败", response.json()["detail"])
        self.assertNotIn(
            "模拟的上游错误详情",
            response.text,
        )
        self.fake_client.close.assert_called_once_with()

    def test_rejects_unknown_mode(self):
        self.payload["mode"] = "unknown"

        with TestClient(api) as client:
            response = client.post(
                "/ask",
                json=self.payload,
            )

        self.assertEqual(response.status_code, 422)
        self.create_client.assert_not_called()

    def test_reports_upstream_status_and_known_codes_without_private_body(self):
        private_text = "sk-private-test-only 原始提示词与论文正文"
        cases = [
            (401, "InvalidApiKey", "密钥"),
            (400, "Arrearage", "账单"),
            (403, "AllocationQuota.FreeTierOnly", "用完即停"),
            (403, "Model.AccessDenied", "权限"),
            (404, "model_not_found", "模型 ID"),
            (400, "invalid_parameter_error", "参数"),
            (404, private_text, "API 地址"),
        ]
        for status, code, expected_hint in cases:
            with self.subTest(status=status, code=code):
                self.fake_client.reset_mock()
                body = {"error": {"code": code, "message": private_text}}
                self.fake_client.responses.create.side_effect = APIStatusError(
                    private_text,
                    response=Response(status, request=self.upstream_request),
                    body=body,
                )
                with self.assertLogs("api", level="WARNING") as logs:
                    with TestClient(api) as client:
                        response = client.post("/ask", json=self.payload)

                self.assertEqual(response.status_code, 502)
                self.assertIn(f"上游 HTTP {status}", response.json()["detail"])
                self.assertIn(expected_hint, response.json()["detail"])
                self.assertNotIn(private_text, response.text)
                self.assertNotIn(private_text, " ".join(logs.output))
                if code != private_text:
                    self.assertIn(code, response.json()["detail"])
                self.fake_client.close.assert_called_once_with()

    def test_reports_connection_failure_without_private_exception(self):
        private_text = "sk-private-test-only 不应回显的连接异常"
        self.fake_client.responses.create.side_effect = APIConnectionError(
            message=private_text,
            request=self.upstream_request,
        )
        with self.assertLogs("api", level="WARNING") as logs:
            with TestClient(api) as client:
                response = client.post("/ask", json=self.payload)

        self.assertEqual(response.status_code, 502)
        self.assertIn("无法连接", response.json()["detail"])
        self.assertNotIn(private_text, response.text)
        self.assertNotIn(private_text, " ".join(logs.output))
        self.fake_client.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
