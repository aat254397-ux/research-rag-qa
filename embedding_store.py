import json
import math
from pathlib import Path

from embedding_retrieval import DIMENSION, ENCODER_SPEC, corpus_fingerprint
from index_store import validate_indexed_chunks, write_json_atomically


FORMAT_VERSION = 1
INDEX_TYPE = "e5-v1"


def validate_embedding_index(retrieval_index):
    if not isinstance(retrieval_index, dict):
        raise ValueError("embedding index must be an object")
    if retrieval_index.get("encoder_spec") != ENCODER_SPEC:
        raise ValueError("embedding encoder configuration is incompatible")
    chunks = retrieval_index.get("chunks")
    validate_indexed_chunks(chunks, DIMENSION, allow_negative=True)
    if retrieval_index.get("source_fingerprint") != corpus_fingerprint(chunks):
        raise ValueError("embedding source fingerprint does not match the chunks")
    for chunk in chunks:
        squared_norm = sum(value * value for value in chunk["vector"])
        if not math.isclose(squared_norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError("E5 chunk vectors must be normalized and non-zero")


def save_embedding_index(retrieval_index, file_path):
    validate_embedding_index(retrieval_index)
    write_json_atomically({
        "format_version": FORMAT_VERSION,
        "index_type": INDEX_TYPE,
        "index": retrieval_index,
    }, file_path)


def load_embedding_index(file_path):
    with Path(file_path).open("r", encoding="utf-8") as snapshot_file:
        payload = json.load(snapshot_file)
    if (
        not isinstance(payload, dict)
        or type(payload.get("format_version")) is not int
        or payload["format_version"] != FORMAT_VERSION
        or payload.get("index_type") != INDEX_TYPE
    ):
        raise ValueError("unsupported embedding index snapshot")
    retrieval_index = payload.get("index")
    validate_embedding_index(retrieval_index)
    return retrieval_index
