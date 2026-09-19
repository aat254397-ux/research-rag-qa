import unittest

from prompt_builder import build_context, build_prompt


class TestPromptBuilder(unittest.TestCase):
    def setUp(self):
        self.retrieval_results = [
            {
                "source": "python_venv.txt",
                "path": "sample_docs/python_venv.txt",
                "chunk_id": 1,
                "start": 0,
                "end": 23,
                "text": (
                    "Python 虚拟环境用于隔离"
                    "不同项目的依赖。"
                ),
                "vector": [1.0, 0.5],
                "score": 0.92,
            },
            {
                "source": "python_venv.txt",
                "path": "sample_docs/python_venv.txt",
                "chunk_id": 2,
                "start": 23,
                "end": 44,
                "text": (
                    "激活环境并确认 Python "
                    "解释器路径。"
                ),
                "vector": [0.8, 0.3],
                "score": 0.53,
            },
        ]

    def test_builds_context_with_source_metadata(self):
        context = build_context(
            self.retrieval_results
        )

        self.assertIn(
            "来源：python_venv.txt",
            context,
        )
        self.assertIn(
            "切片：1",
            context,
        )
        self.assertIn(
            "位置：0-23",
            context,
        )

    def test_combines_multiple_chunk_texts(self):
        context = build_context(
            self.retrieval_results
        )

        self.assertIn(
            self.retrieval_results[0]["text"],
            context,
        )
        self.assertIn(
            self.retrieval_results[1]["text"],
            context,
        )

    def test_excludes_retrieval_calculation_data(self):
        context = build_context(
            self.retrieval_results
        )

        self.assertNotIn("vector", context)
        self.assertNotIn("score", context)
        self.assertNotIn("0.92", context)

    def test_builds_prompt_with_context_and_question(self):
        context = build_context(
            self.retrieval_results
        )
        question_text = (
            "如何隔离不同 Python 项目的依赖？"
        )

        prompt = build_prompt(
            question_text,
            context,
        )

        self.assertIn(
            "请只根据参考资料回答问题",
            prompt,
        )
        self.assertIn(context, prompt)
        self.assertIn(question_text, prompt)
        self.assertTrue(
            prompt.endswith("回答：")
        )

    def test_rejects_empty_retrieval_results(self):
        with self.assertRaises(ValueError):
            build_context([])

    def test_rejects_empty_prompt_inputs(self):
        with self.assertRaises(ValueError):
            build_prompt(
                "   ",
                "有效参考资料",
            )

        with self.assertRaises(ValueError):
            build_prompt(
                "有效问题",
                "\n",
            )


if __name__ == "__main__":
    unittest.main()
