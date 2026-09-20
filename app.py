from document_loader import load_document
from prompt_builder import build_context, build_prompt
from retrieval import (
    build_retrieval_index,
    retrieve_relevant_chunks,
)
from text_chunker import chunk_document


def build_document_qa_index(
    document_paths,
    chunk_size=80,
    overlap=20,
):
    knowledge_chunks = []

    for document_path in document_paths:
        loaded_document = load_document(
            document_path
        )

        document_chunks = chunk_document(
            loaded_document,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        knowledge_chunks.extend(
            document_chunks
        )

    return build_retrieval_index(
        knowledge_chunks
    )


def build_document_qa_prompt(
    question_text,
    retrieval_index,
    top_k=2,
    min_score=0.1,
):
    retrieval_results = retrieve_relevant_chunks(
        question_text,
        retrieval_index,
        top_k=top_k,
        min_score=min_score,
    )

    context = build_context(
        retrieval_results
    )

    return build_prompt(
        question_text,
        context,
    )


if __name__ == "__main__":
    document_paths = [
        "sample_docs/python_venv.txt",
        "sample_docs/rag_notes.md",
    ]

    retrieval_index = build_document_qa_index(
        document_paths,
        chunk_size=80,
        overlap=20,
    )

    question_text = (
        "如何 隔离 Python 项目 依赖"
    )

    prompt = build_document_qa_prompt(
        question_text,
        retrieval_index,
        top_k=2,
        min_score=0.1,
    )

    print(prompt)
