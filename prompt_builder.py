def build_context(retrieval_results):
    if not retrieval_results:
        raise ValueError(
            "retrieval_results must not be empty"
        )

    context_sections = []

    for retrieval_result in retrieval_results:
        source = retrieval_result["source"]
        chunk_id = retrieval_result["chunk_id"]
        start = retrieval_result["start"]
        end = retrieval_result["end"]
        text = retrieval_result["text"].strip()

        context_section = (
            f"[来源：{source} | "
            f"切片：{chunk_id} | "
            f"位置：{start}-{end}]\n"
            f"{text}"
        )

        context_sections.append(
            context_section
        )

    return "\n\n".join(context_sections)


def build_prompt(
    question_text,
    context,
):
    if not question_text.strip():
        raise ValueError(
            "question_text must not be empty"
        )

    if not context.strip():
        raise ValueError(
            "context must not be empty"
        )

    return (
        "你是一个文档问答助手。\n"
        "请只根据参考资料回答问题。\n"
        "如果参考资料不足以回答，"
        "请明确说明“根据现有资料无法回答”。\n"
        "回答时请简洁，并注明使用的资料来源。\n\n"
        "参考资料：\n"
        f"{context.strip()}\n\n"
        "用户问题：\n"
        f"{question_text.strip()}\n\n"
        "回答："
    )


if __name__ == "__main__":
    sample_retrieval_results = [
        {
            "source": "python_venv.txt",
            "path": "sample_docs/python_venv.txt",
            "chunk_id": 1,
            "start": 0,
            "end": 23,
            "text": (
                "Python 虚拟环境用于隔离"
                "不同项目的依赖。"
            ),
            "vector": [1.0, 0.5],
            "score": 0.92,
        },
        {
            "source": "python_venv.txt",
            "path": "sample_docs/python_venv.txt",
            "chunk_id": 2,
            "start": 23,
            "end": 44,
            "text": (
                "激活环境并确认 Python "
                "解释器路径。"
            ),
            "vector": [0.8, 0.3],
            "score": 0.53,
        },
    ]

    question_text = (
        "如何隔离不同 Python 项目的依赖？"
    )

    context = build_context(
        sample_retrieval_results
    )

    prompt = build_prompt(
        question_text,
        context,
    )

    print(prompt)
