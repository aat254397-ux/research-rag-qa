import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from model_client import generate_answer


class TestModelClient(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.client.responses.create.return_value = SimpleNamespace(
            output_text="  这是模拟回答。\n"
        )

    def test_passes_prompt_and_model_to_client(self):
        prompt = "请根据参考资料回答问题。"

        generate_answer(
            prompt=prompt,
            client=self.client,
            model="test-model",
        )

        self.client.responses.create.assert_called_once_with(
            model="test-model",
            input=prompt,
        )

    def test_returns_stripped_answer(self):
        answer = generate_answer(
            prompt="有效问题",
            client=self.client,
            model="test-model",
        )

        self.assertEqual(answer, "这是模拟回答。")

    def test_rejects_empty_prompt(self):
        with self.assertRaises(ValueError):
            generate_answer(
                prompt=" \n ",
                client=self.client,
                model="test-model",
            )

        self.client.responses.create.assert_not_called()

    def test_rejects_empty_model(self):
        with self.assertRaises(ValueError):
            generate_answer(
                prompt="有效问题",
                client=self.client,
                model="  ",
            )

        self.client.responses.create.assert_not_called()

    def test_rejects_empty_answer(self):
        self.client.responses.create.return_value = SimpleNamespace(
            output_text=" \n "
        )

        with self.assertRaises(ValueError):
            generate_answer(
                prompt="有效问题",
                client=self.client,
                model="test-model",
            )

    def test_propagates_client_error(self):
        self.client.responses.create.side_effect = RuntimeError(
            "模拟服务不可用"
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "模拟服务不可用",
        ):
            generate_answer(
                prompt="有效问题",
                client=self.client,
                model="test-model",
            )


if __name__ == "__main__":
    unittest.main()
