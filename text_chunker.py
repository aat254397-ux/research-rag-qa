from document_loader import load_document


def validate_chunk_settings(
    chunk_size,
    overlap,
):
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )

    if overlap < 0:
        raise ValueError(
            "overlap must not be negative"
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )


def chunk_document(
    loaded_document,
    chunk_size,
    overlap,
):
    validate_chunk_settings(
        chunk_size,
        overlap,
    )

    document_text = loaded_document["text"]
    source_name = loaded_document["source"]
    source_path = loaded_document["path"]

    document_chunks = []
    start = 0
    chunk_id = 1

    while start < len(document_text):
        end = min(
            start + chunk_size,
            len(document_text),
        )

        chunk_text = document_text[start:end]

        document_chunk = {
            "chunk_id": chunk_id,
            "source": source_name,
            "path": source_path,
            "text": chunk_text,
            "start": start,
            "end": end,
        }

        if "page" in loaded_document:
            document_chunk["page"] = loaded_document["page"]

        document_chunks.append(document_chunk)

        if end >= len(document_text):
            break

        start = end - overlap
        chunk_id += 1

    return document_chunks


if __name__ == "__main__":
    loaded_document = load_document(
        "sample_docs/python_venv.txt"
    )

    document_chunks = chunk_document(
        loaded_document,
        chunk_size=40,
        overlap=10,
    )

    print(
        "文档来源：",
        loaded_document["source"],
    )
    print(
        "切片数量：",
        len(document_chunks),
    )

    for document_chunk in document_chunks:
        print("-" * 40)
        print(
            "切片编号：",
            document_chunk["chunk_id"],
        )
        print(
            "字符范围：",
            document_chunk["start"],
            "到",
            document_chunk["end"],
        )
        print(
            "切片内容：",
            document_chunk["text"],
        )
