import hashlib
import json
import math
from pathlib import Path
from threading import Lock

from retrieval import cosine_similarity


MODEL_NAME = "intfloat/multilingual-e5-small"
# 固定为本次已下载并跑通 demo 的权重版本，避免恢复索引时模型悄悄变化。
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DIMENSION = 384
MAX_TOKENS = 512
MODEL_CACHE_DIR = Path(__file__).resolve().parent / "data" / "model_cache"
ENCODER_SPEC = {
    "model_name": MODEL_NAME,
    "revision": MODEL_REVISION,
    "dimension": DIMENSION,
    "pooling": "mean",
    "normalized": True,
    "query_prefix": "query: ",
    "passage_prefix": "passage: ",
    "max_tokens": MAX_TOKENS,
    "text_preprocessing": "raw-v1",
}
CHUNK_FIELDS = ("source", "path", "chunk_id", "start", "end", "text", "page")


class EmbeddingUnavailableError(RuntimeError):
    pass


def plain_chunks(indexed_chunks):
    """保留同一批原文和元数据，不携带旧的 TF-IDF 向量或检索分数。"""
    return [
        {field: chunk[field] for field in CHUNK_FIELDS if field in chunk}
        for chunk in indexed_chunks
    ]


def corpus_fingerprint(chunks):
    serialized = json.dumps(
        plain_chunks(chunks),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class E5Encoder:
    """服务内懒加载、复用同一份 CPU 模型；不联网补下载。"""

    def __init__(self):
        self._model = None
        self._lock = Lock()

    @property
    def loaded(self):
        return self._model is not None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                MODEL_NAME,
                revision=MODEL_REVISION,
                device="cpu",
                cache_folder=str(MODEL_CACHE_DIR),
                local_files_only=True,
                trust_remote_code=False,
            )
        except (ImportError, OSError, ValueError) as error:
            raise EmbeddingUnavailableError(
                "无法加载本地 E5 模型，请确认 requirements-embedding.txt 已安装，"
                "且 embedding_demo.py 所用的模型缓存完整。"
            ) from error
        self._model = model

    def _encode(self, texts, prefix, role):
        inputs = [prefix + text for text in texts]
        # 模型加载和推理共用锁，避免并发请求重复加载或同时占用大量 CPU 内存。
        with self._lock:
            self._load_model()
            lengths = self._model.tokenizer(
                inputs,
                truncation=False,
                padding=False,
                return_length=True,
            )["length"]
            for position, length in enumerate(lengths, start=1):
                if length > MAX_TOKENS:
                    raise ValueError(
                        f"{role}第 {position} 条含 {length} 个 token，超过 E5 的 "
                        f"{MAX_TOKENS} 上限（包含前缀和特殊 token）。"
                        "本次未静默截断；请缩短问题或调整切片后重建索引。"
                    )

            return self._model.encode(
                inputs,
                batch_size=8,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
                # 前缀已手动加入，禁止模型另加默认 prompt。
                prompt="",
            ).tolist()

    def encode_passages(self, texts):
        return self._encode(texts, ENCODER_SPEC["passage_prefix"], "资料片段")

    def encode_query(self, text):
        return self._encode([text], ENCODER_SPEC["query_prefix"], "问题")[0]

    def retrieve(self, question_text, retrieval_index, top_k=3, min_score=-1.0):
        if retrieval_index.get("encoder_spec") != ENCODER_SPEC:
            raise ValueError("E5 索引与当前编码配置不兼容，请重新构建。")
        query_vector = self.encode_query(question_text)
        return retrieve_embedding_chunks(
            query_vector, retrieval_index, top_k=top_k, min_score=min_score
        )


def build_embedding_index(knowledge_chunks, encoder):
    if not knowledge_chunks:
        raise ValueError("knowledge_chunks must not be empty")
    chunks = plain_chunks(knowledge_chunks)
    vectors = encoder.encode_passages([chunk["text"] for chunk in chunks])
    if len(vectors) != len(chunks):
        raise ValueError("E5 vector count does not match chunk count")
    return {
        "encoder_spec": dict(ENCODER_SPEC),
        "source_fingerprint": corpus_fingerprint(chunks),
        "chunks": [
            {**chunk, "vector": vector}
            for chunk, vector in zip(chunks, vectors)
        ],
    }


def retrieve_embedding_chunks(
    query_vector, retrieval_index, top_k=3, min_score=-1.0
):
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if not -1.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between -1 and 1")
    if (
        len(query_vector) != DIMENSION
        or not all(
            type(value) in (int, float) and math.isfinite(value)
            for value in query_vector
        )
        or not any(query_vector)
    ):
        raise ValueError("E5 query vector has invalid values or dimension")

    results = []
    for chunk in retrieval_index["chunks"]:
        # 消除浮点误差导致的 1.000000...，分数仍不是概率。
        score = max(-1.0, min(1.0, cosine_similarity(query_vector, chunk["vector"])))
        if score >= min_score:
            results.append({**chunk, "score": score})
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]
