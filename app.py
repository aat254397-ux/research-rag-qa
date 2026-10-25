from pathlib import Path

from document_loader import load_document
from model_client import generate_answer
from pdf_loader import load_pdf_pages
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
    pdf_chunk_size=1000,
    pdf_overlap=200,
):
    knowledge_chunks = []

    for document_path in document_paths:
        if Path(document_path).suffix.lower() == ".pdf":
            loaded_units = load_pdf_pages(document_path)
            unit_chunk_size = pdf_chunk_size
            unit_overlap = pdf_overlap
        else:
            loaded_units = [load_document(document_path)]
            unit_chunk_size = chunk_size
            unit_overlap = overlap

        # PDF 每页分别切片，但编号在同一文件内连续。
        next_chunk_id = 1

        for loaded_unit in loaded_units:
            if not loaded_unit["text"].strip():
                continue

            document_chunks = chunk_document(
                loaded_unit,
                chunk_size=unit_chunk_size,
                overlap=unit_overlap,
            )

            for document_chunk in document_chunks:
                document_chunk["chunk_id"] = next_chunk_id
                next_chunk_id += 1

            knowledge_chunks.extend(document_chunks)

    return build_retrieval_index(knowledge_chunks)


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

    context = build_context(retrieval_results)

    return build_prompt(question_text, context)


def search_document_references(
    question_text,
    retrieval_index,
    top_k=2,
    min_score=0.1,
    retrieval_function=retrieve_relevant_chunks,
    candidate_k=10,
    reranker=None,
):
    """检索、可选重排并整理参考资料，不调用生成模型。"""
    if reranker is not None and candidate_k < top_k:
        raise ValueError("candidate_k must be greater than or equal to top_k")
    retrieval_results = retrieval_function(
        question_text,
        retrieval_index,
        top_k=candidate_k if reranker is not None else top_k,
        min_score=min_score,
    )

    references = []

    for result in retrieval_results:
        reference = {
            "source": result["source"],
            "path": result["path"],
            "chunk_id": result["chunk_id"],
            "start": result["start"],
            "end": result["end"],
            "text": result["text"],
            "score": result["score"],
        }

        if "page" in result:
            reference["page"] = result["page"]

        references.append(reference)

    if reranker is not None:
        references = reranker.rerank(question_text, references)[:top_k]
        references = [
            {**reference, "rank": rank}
            for rank, reference in enumerate(references, start=1)
        ]

    return references


def answer_document_question(
    question_text,
    retrieval_index,
    client,
    model,
    top_k=2,
    min_score=0.1,
    retrieval_function=retrieve_relevant_chunks,
    candidate_k=10,
    reranker=None,
):
    # 1. 与纯检索接口共用同一个函数，确保两处选取资料的方式一致。
    references = search_document_references(
        question_text,
        retrieval_index,
        top_k=top_k,
        min_score=min_score,
        retrieval_function=retrieval_function,
        candidate_k=candidate_k,
        reranker=reranker,
    )

    # 2. 将实际选中的参考资料和问题组成 Prompt。
    context = build_context(references)
    prompt = build_prompt(question_text, context)

    # 3. 生成答案，并连同实际使用的参考资料返回。
    answer = generate_answer(
        prompt=prompt,
        client=client,
        model=model,
    )

    return {
        "question": question_text,
        "answer": answer,
        "references": references,
    }


if __name__ == "__main__":
    from types import SimpleNamespace
    from unittest.mock import Mock

    document_paths = [
        "sample_docs/python_venv.txt",
        "sample_docs/rag_notes.md",
    ]

    retrieval_index = build_document_qa_index(
        document_paths,
        chunk_size=80,
        overlap=20,
    )

    # 演示用假客户端，不发送网络请求
    demo_client = Mock()
    demo_client.responses.create.return_value = SimpleNamespace(
        output_text=(
            "可以为不同的 Python 项目创建独立的虚拟环境，"
            "分别管理各自的依赖。"
        )
    )

    result = answer_document_question(
        question_text="如何 隔离 Python 项目 依赖",
        retrieval_index=retrieval_index,
        client=demo_client,
        model="test-model",
        top_k=2,
        min_score=0.1,
    )

    print("运行模式：模拟客户端，答案为预设内容")
    print("问题：", result["question"])
    print("答案：", result["answer"])
    print("\n提供给模型的参考资料：")

    for reference in result["references"]:
        print(
            f"- 来源：{reference['source']} | "
            f"切片：{reference['chunk_id']} | "
            f"位置：{reference['start']}-{reference['end']}"
        )
        print(f"  路径：{reference['path']}")
        print(f"  正文：{reference['text']}")
