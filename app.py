from document_loader import load_document
from model_client import generate_answer
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
        loaded_document = load_document(document_path)

        document_chunks = chunk_document(
            loaded_document,
            chunk_size=chunk_size,
            overlap=overlap,
        )

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


def answer_document_question(
    question_text,
    retrieval_index,
    client,
    model,
    top_k=2,
    min_score=0.1,
):
    # 1. 找到相关资料
    retrieval_results = retrieve_relevant_chunks(
        question_text,
        retrieval_index,
        top_k=top_k,
        min_score=min_score,
    )

    # 2. 用这些资料和问题组装 Prompt
    context = build_context(retrieval_results)
    prompt = build_prompt(question_text, context)

    # 3. 调用客户端，取得答案
    answer = generate_answer(
        prompt=prompt,
        client=client,
        model=model,
    )

    # 4. 保留本次提供给模型的资料，方便追踪来源
    references = []

    for result in retrieval_results:
        references.append({
            "source": result["source"],
            "path": result["path"],
            "chunk_id": result["chunk_id"],
            "start": result["start"],
            "end": result["end"],
            "text": result["text"],
            "score": result["score"],
        })

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
