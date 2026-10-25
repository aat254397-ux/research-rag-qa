import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


FORMAT_VERSION = 1
# 分词、权重计算或向量含义改变时，需要升级此标记。
INDEX_TYPE = "tfidf-v1"


def _is_finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_retrieval_index(retrieval_index):
    """检查快照能否供当前检索代码使用，不重新计算索引。"""
    if not isinstance(retrieval_index, dict):
        raise ValueError("index must be an object")

    vocabulary = retrieval_index.get("vocabulary")
    idf_weights = retrieval_index.get("idf_weights")
    chunks = retrieval_index.get("chunks")

    if (
        not isinstance(vocabulary, list)
        or not vocabulary
        or not all(isinstance(term, str) and term for term in vocabulary)
        or len(set(vocabulary)) != len(vocabulary)
    ):
        raise ValueError("index vocabulary must contain unique non-empty terms")

    if (
        not isinstance(idf_weights, dict)
        or set(idf_weights) != set(vocabulary)
        or not all(
            _is_finite_number(weight) and weight > 0
            for weight in idf_weights.values()
        )
    ):
        raise ValueError("index IDF weights must match the vocabulary")

    validate_indexed_chunks(chunks, len(vocabulary))


def validate_indexed_chunks(chunks, dimension, allow_negative=False):
    """两类索引共用来源校验，但 TF-IDF 与 E5 的数值范围不同。"""
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("index chunks must not be empty")

    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("each indexed chunk must be an object")

        for field in ("source", "path", "text"):
            value = chunk.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"chunk {field} must not be empty")

        chunk_id = chunk.get("chunk_id")
        start = chunk.get("start")
        end = chunk.get("end")
        if (
            type(chunk_id) is not int
            or chunk_id < 1
            or type(start) is not int
            or type(end) is not int
            or not 0 <= start < end
        ):
            raise ValueError("chunk ID or character positions are invalid")

        if "page" in chunk:
            if type(chunk["page"]) is not int or chunk["page"] < 1:
                raise ValueError("PDF page must be a positive integer")

        vector = chunk.get("vector")
        if (
            not isinstance(vector, list)
            or len(vector) != dimension
            or not all(
                _is_finite_number(value) and (allow_negative or value >= 0)
                for value in vector
            )
        ):
            raise ValueError("chunk vector has invalid values or dimension")


def save_retrieval_index(retrieval_index, file_path):
    """先完整写入同目录临时文件，成功后替换正式快照。"""
    validate_retrieval_index(retrieval_index)
    payload = {
        "format_version": FORMAT_VERSION,
        "index_type": INDEX_TYPE,
        "index": retrieval_index,
    }
    write_json_atomically(payload, file_path)


def write_json_atomically(payload, file_path):
    """保存完整 JSON 文件；具体索引类型的校验由调用方负责。"""
    snapshot_path = Path(file_path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=snapshot_path.parent,
            prefix=f".{snapshot_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(
                payload,
                temporary_file,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        # Windows 下先关闭临时文件，再替换目标文件。
        os.replace(temporary_path, snapshot_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_retrieval_index(file_path):
    snapshot_path = Path(file_path)
    with snapshot_path.open("r", encoding="utf-8") as snapshot_file:
        payload = json.load(snapshot_file)

    if not isinstance(payload, dict):
        raise ValueError("index snapshot must be an object")
    if (
        type(payload.get("format_version")) is not int
        or payload["format_version"] != FORMAT_VERSION
        or payload.get("index_type") != INDEX_TYPE
    ):
        raise ValueError("unsupported index snapshot version or index type")

    retrieval_index = payload.get("index")
    validate_retrieval_index(retrieval_index)
    return retrieval_index
