from pathlib import Path

from retrieval import cosine_similarity


MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_CACHE_DIR = Path(__file__).resolve().parent / "data" / "model_cache"

QUESTIONS = [
    "Which variables are used to construct the MJO index?",
    "构建 MJO 指数使用了哪些变量？",
]

# 人工编写的短例句，仅用于观察模型输出，不代表真实论文检索评测。
PASSAGES = [
    {
        "id": "A",
        "text": (
            "The MJO index is constructed from outgoing longwave radiation "
            "(OLR), 850-hPa zonal wind, and 200-hPa zonal wind."
        ),
    },
    {
        "id": "B",
        "text": (
            "The Madden-Julian Oscillation (MJO) is a large-scale pattern "
            "of tropical atmospheric variability. Its convection moves "
            "eastward and influences rainfall and weather."
        ),
    },
    {
        "id": "C",
        "text": (
            "Python virtual environments isolate project dependencies. "
            "Different projects can install different package versions."
        ),
    },
]


def main():
    # 只有运行本例时才需要此可选依赖；不会被现有 API 自动加载。
    from sentence_transformers import SentenceTransformer

    print("准备加载 Embedding 模型：", MODEL_NAME)
    print("模型文件缓存目录：", MODEL_CACHE_DIR)
    print("首次运行可能联网下载模型；编码计算在本机 CPU 上完成。", flush=True)

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        cache_folder=str(MODEL_CACHE_DIR),
    )

    # E5 要求区分问题与候选段落，中文输入同样使用这两个英文前缀。
    query_inputs = [f"query: {question}" for question in QUESTIONS]
    passage_inputs = [f"passage: {passage['text']}" for passage in PASSAGES]

    print("模型已加载，正在把问题和段落转换成向量……", flush=True)
    query_vectors = model.encode(
        query_inputs,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    passage_vectors = model.encode(
        passage_inputs,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    print("\n问题向量的形状：", query_vectors.shape)
    print("段落向量的形状：", passage_vectors.shape)
    print(
        "第一个问题向量的前 8 个数：",
        [round(float(value), 4) for value in query_vectors[0][:8]],
    )

    for question, query_vector in zip(QUESTIONS, query_vectors):
        ranked_passages = []

        for passage, passage_vector in zip(PASSAGES, passage_vectors):
            # 转成普通 Python 列表，复用之前 TF-IDF 使用的余弦相似度函数。
            score = cosine_similarity(
                query_vector.tolist(),
                passage_vector.tolist(),
            )
            ranked_passages.append({
                "id": passage["id"],
                "text": passage["text"],
                "score": score,
            })

        ranked_passages.sort(key=lambda item: item["score"], reverse=True)

        print("\n问题：", question)
        for rank, passage in enumerate(ranked_passages, start=1):
            print(
                f"第 {rank} 名 | 段落 {passage['id']} | "
                f"相似度：{passage['score']:.4f}"
            )
            print("  原文：", passage["text"])

    print("\n以上分数是真实模型计算结果，不是正确率；本例不预设排名。")
    print("这里只比较全部三个段落，没有套用 TF-IDF 的相似度阈值。")
    print("本例未读取论文、修改索引或调用 DeepSeek，也没有训练模型。")


if __name__ == "__main__":
    main()
