import unittest
from copy import deepcopy

from pydantic import ValidationError

from api import SearchRequest
from reranking import CrossEncoderReranker, rank_candidates_by_scores


class TestReranking(unittest.TestCase):
    def test_keeps_original_scores_ranks_and_metadata_without_mutating_input(self):
        candidates = [
            {"text": "background", "score": 0.83, "chunk_id": 66, "page": 15},
            {"text": "answer", "score": 0.82, "chunk_id": 1, "page": 1},
        ]
        original = deepcopy(candidates)

        ranked = rank_candidates_by_scores(candidates, [-1.0, 2.5])

        self.assertEqual([item["chunk_id"] for item in ranked], [1, 66])
        self.assertEqual(ranked[0]["retrieval_rank"], 2)
        self.assertEqual(ranked[0]["score"], 0.82)
        self.assertEqual(ranked[0]["rerank_score"], 2.5)
        self.assertEqual(ranked[0]["page"], 1)
        self.assertEqual(candidates, original)

    def test_rejects_mismatched_or_non_finite_scores(self):
        for scores in ([], [float("nan")], [float("inf")]):
            with self.subTest(scores=scores):
                with self.assertRaises(ValueError):
                    rank_candidates_by_scores([{"text": "sample"}], scores)

    def test_empty_candidates_do_not_load_model(self):
        reranker = CrossEncoderReranker()
        self.assertEqual(reranker.rerank("question", []), [])
        self.assertFalse(reranker.loaded)

    def test_accepts_two_stage_retrieval_parameters(self):
        request = SearchRequest(
            question="构建 MJO 指数使用了哪些变量？",
            retriever="embedding", rerank=True, candidate_k=10, top_k=3,
        )
        self.assertTrue(request.rerank)
        self.assertEqual(request.candidate_k, 10)
        self.assertEqual(request.top_k, 3)

    def test_rejects_inconsistent_or_oversized_candidate_settings(self):
        invalid_settings = [
            {"candidate_k": 2, "top_k": 3},
            {"candidate_k": 0},
            {"candidate_k": 51},
            {"retriever": "tfidf"},
        ]
        for settings in invalid_settings:
            with self.subTest(settings=settings):
                payload = {
                    "question": "MJO", "retriever": "embedding", "rerank": True,
                    **settings,
                }
                with self.assertRaises(ValidationError):
                    SearchRequest(**payload)

    def test_reranking_is_optional_and_disabled_by_default(self):
        request = SearchRequest(question="Python", top_k=3, candidate_k=1)
        self.assertFalse(request.rerank)
        self.assertEqual(request.retriever, "tfidf")


if __name__ == "__main__":
    unittest.main()
