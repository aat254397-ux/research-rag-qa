import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

from app import answer_document_question
from prompt_builder import build_context, build_prompt
from retrieval import build_retrieval_index


class TestAnswerPipeline(unittest.TestCase):
    def setUp(self):
        # 使用固定的小资料，避免测试依赖真实文档内容
        self.chunks = [
            {
                "source": "python.txt",
                "path": "sample_docs/python.txt",
                "chunk_id": 1,
                "start": 0,
                "end": 18,
                "text": "python environment",
            },
            {
                "source": "ocean.txt",
                "path": "sample_docs/ocean.txt",
                "chunk_id": 1,
                "start": 0,
                "end": 17,
                "text": "ocean temperature",
            },
        ]

        self.index = build_retrieval_index(self.chunks)

        self.client = Mock()
        self.client.responses.create.return_value = SimpleNamespace(
            output_text="  使用独立的 Python 环境。\n"
        )

    def test_returns_answer_and_matching_reference(self):
        question = "python environment"
        original_index = deepcopy(self.index)

        result = answer_document_question(
            question_text=question,
            retrieval_index=self.index,
            client=self.client,
            model="test-model",
            top_k=1,
            min_score=0.1,
        )

        # 检查返回的问题和答案
        self.assertEqual(result["question"], question)
        self.assertEqual(
            result["answer"],
            "使用独立的 Python 环境。",
        )

        # 检查返回的参考资料是否正确、完整
        self.assertEqual(len(result["references"]), 1)
        reference = result["references"][0]

        for field, value in self.chunks[0].items():
            self.assertEqual(reference[field], value)

        self.assertGreaterEqual(reference["score"], 0.1)
        self.assertNotIn("vector", reference)

        # 检查实际传给客户端的 Prompt
        expected_context = build_context([self.chunks[0]])
        expected_prompt = build_prompt(
            question,
            expected_context,
        )

        self.client.responses.create.assert_called_once_with(
            model="test-model",
            input=expected_prompt,
        )

        # 回答流程不应该修改原索引
        self.assertEqual(self.index, original_index)

    def test_does_not_call_client_for_unknown_question(self):
        with self.assertRaises(ValueError):
            answer_document_question(
                question_text="astronomy",
                retrieval_index=self.index,
                client=self.client,
                model="test-model",
            )

        self.client.responses.create.assert_not_called()

    def test_propagates_model_service_error(self):
        self.client.responses.create.side_effect = RuntimeError(
            "模拟服务不可用"
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "模拟服务不可用",
        ):
            answer_document_question(
                question_text="python environment",
                retrieval_index=self.index,
                client=self.client,
                model="test-model",
            )

        self.client.responses.create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
