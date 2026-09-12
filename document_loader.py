from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
}


def load_document(file_path):
    document_path = Path(file_path)

    if not document_path.exists():
        raise FileNotFoundError(
            f"document does not exist: {document_path}"
        )

    if not document_path.is_file():
        raise ValueError(
            f"path is not a file: {document_path}"
        )

    document_extension = document_path.suffix.lower()

    if document_extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"unsupported document type: {document_extension}"
        )

    document_text = document_path.read_text(
        encoding="utf-8"
    )

    if not document_text.strip():
        raise ValueError("document must not be empty")

    return {
        "source": document_path.name,
        "path": str(document_path),
        "text": document_text,
    }


if __name__ == "__main__":
    sample_document_paths = [
        "sample_docs/python_venv.txt",
        "sample_docs/rag_notes.md",
    ]

    for sample_document_path in sample_document_paths:
        loaded_document = load_document(
            sample_document_path
        )

        print("来源：", loaded_document["source"])
        print("路径：", loaded_document["path"])
        print("正文：")
        print(loaded_document["text"])
        print("-" * 40)
