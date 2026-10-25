"""API 与独立例子共用的 Cross-Encoder 重排组件。"""

import math
from pathlib import Path
from threading import Lock


MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MODEL_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
MODEL_CACHE_DIR = Path(__file__).resolve().parent / "data" / "model_cache"
MAX_PAIR_TOKENS = 512
DEFAULT_BATCH_SIZE = 4


class RerankerUnavailableError(RuntimeError):
    pass


def rank_candidates_by_scores(candidates, scores):
    """保留检索分数与原名次，按新的重排分数排序；不修改输入。"""
    if len(scores) != len(candidates) or not all(
        type(score) in (int, float) and math.isfinite(score) for score in scores
    ):
        raise ValueError("重排分数数量或数值异常，本次不输出排名")
    ranked = [
        {
            **candidate,
            "retrieval_rank": candidate.get("retrieval_rank", candidate.get("rank", position)),
            "rerank_score": float(score),
        }
        for position, (candidate, score) in enumerate(zip(candidates, scores), start=1)
    ]
    return sorted(ranked, key=lambda item: item["rerank_score"], reverse=True)


def rerank_candidates(model, question, candidates, batch_size=DEFAULT_BATCH_SIZE,
                      show_progress_bar=False):
    # 不加 E5 前缀，问题和候选作为文本对联合输入。
    pairs = [(question, candidate["text"]) for candidate in candidates]
    lengths = model.tokenizer(
        [question] * len(candidates),
        [candidate["text"] for candidate in candidates],
        padding=False,
        truncation=False,
        return_length=True,
    )["length"]
    for position, (candidate, length) in enumerate(zip(candidates, lengths), start=1):
        if length > MAX_PAIR_TOKENS:
            original_rank = candidate.get("retrieval_rank", candidate.get("rank", position))
            raise ValueError(
                f"原第 {original_rank} 名（切片 {candidate['chunk_id']}）与问题"
                f"合计 {length} 个 token，超过 {MAX_PAIR_TOKENS} 上限。"
                "本次未静默截断，也未修改输入或索引。"
            )
    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        prompt="",
    ).reshape(-1).tolist()
    return rank_candidates_by_scores(candidates, scores)


class CrossEncoderReranker:
    """懒加载同一份 CPU 模型；API 默认只读缓存，演示脚本可允许下载。"""

    def __init__(self, local_files_only=True, batch_size=DEFAULT_BATCH_SIZE):
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")
        self.local_files_only = local_files_only
        self.batch_size = batch_size
        self._model = None
        self._lock = Lock()

    @property
    def loaded(self):
        return self._model is not None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            from torch.nn import Identity

            self._model = CrossEncoder(
                MODEL_NAME,
                revision=MODEL_REVISION,
                device="cpu",
                cache_folder=str(MODEL_CACHE_DIR),
                local_files_only=self.local_files_only,
                trust_remote_code=False,
                model_kwargs={"use_safetensors": True},
                max_length=MAX_PAIR_TOKENS,
                activation_fn=Identity(),
            )
        except (ImportError, OSError, ValueError) as error:
            raise RerankerUnavailableError(
                "无法加载重排模型。请确认 Embedding 依赖已安装，"
                "并在项目终端运行 python rerank_demo.py 下载模型后重试。"
            ) from error

    def load(self):
        with self._lock:
            self._load_model()

    def rerank(self, question, candidates, show_progress_bar=False):
        if not candidates:
            return []
        # 避免并发重复加载，或多个 CPU 推理同时占用大量内存。
        with self._lock:
            self._load_model()
            return rerank_candidates(
                self._model, question, candidates,
                batch_size=self.batch_size,
                show_progress_bar=show_progress_bar,
            )
