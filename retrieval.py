import math
import string
from document_loader import load_document
from text_chunker import chunk_document


def tokenize_text(text):
    normalized_text = text.lower()

    punctuation_marks = (
        string.punctuation
        + "，。！？；：、（）【】《》“”‘’"
    )

    for punctuation_mark in punctuation_marks:
        normalized_text = normalized_text.replace(
            punctuation_mark,
            " ",
        )

    return normalized_text.split()


def build_vocabulary(knowledge_chunks):
    vocabulary_terms = set()

    for knowledge_chunk in knowledge_chunks:
        chunk_words = tokenize_text(
            knowledge_chunk["text"]
        )
        vocabulary_terms.update(chunk_words)

    return sorted(vocabulary_terms)


def document_frequency(
    vocabulary_term,
    knowledge_chunks,
):
    chunk_count = 0

    for knowledge_chunk in knowledge_chunks:
        chunk_words = tokenize_text(
            knowledge_chunk["text"]
        )

        if vocabulary_term.lower() in chunk_words:
            chunk_count += 1

    return chunk_count


def inverse_document_frequency(
    vocabulary_term,
    knowledge_chunks,
):
    if not knowledge_chunks:
        raise ValueError(
            "knowledge_chunks must not be empty"
        )

    total_chunk_count = len(knowledge_chunks)

    term_document_frequency = document_frequency(
        vocabulary_term,
        knowledge_chunks,
    )

    return math.log(
        (total_chunk_count + 1)
        / (term_document_frequency + 1)
    ) + 1


def build_idf_weights(
    vocabulary,
    knowledge_chunks,
):
    idf_weights = {}

    for vocabulary_term in vocabulary:
        idf_weights[vocabulary_term] = (
            inverse_document_frequency(
                vocabulary_term,
                knowledge_chunks,
            )
        )

    return idf_weights


def text_to_tfidf_vector(
    text,
    vocabulary,
    idf_weights,
):
    text_words = tokenize_text(text)
    text_vector = []

    for vocabulary_term in vocabulary:
        term_frequency = text_words.count(
            vocabulary_term
        )

        tfidf_value = (
            term_frequency
            * idf_weights[vocabulary_term]
        )

        text_vector.append(tfidf_value)

    return text_vector


def build_retrieval_index(knowledge_chunks):
    if not knowledge_chunks:
        raise ValueError(
            "knowledge_chunks must not be empty"
        )

    vocabulary = build_vocabulary(
        knowledge_chunks
    )

    if not vocabulary:
        raise ValueError(
            "vocabulary must not be empty"
        )

    idf_weights = build_idf_weights(
        vocabulary,
        knowledge_chunks,
    )

    indexed_chunks = []

    for knowledge_chunk in knowledge_chunks:
        chunk_vector = text_to_tfidf_vector(
            knowledge_chunk["text"],
            vocabulary,
            idf_weights,
        )

        indexed_chunk = knowledge_chunk.copy()
        indexed_chunk["vector"] = chunk_vector

        indexed_chunks.append(indexed_chunk)

    return {
        "vocabulary": vocabulary,
        "idf_weights": idf_weights,
        "chunks": indexed_chunks,
    }


def dot_product(vector_a, vector_b):
    if len(vector_a) != len(vector_b):
        raise ValueError(
            "vectors must have the same length"
        )

    product = 0.0

    for value_a, value_b in zip(
        vector_a,
        vector_b,
    ):
        product += value_a * value_b

    return product


def vector_magnitude(vector):
    return math.sqrt(
        sum(
            vector_value ** 2
            for vector_value in vector
        )
    )


def cosine_similarity(vector_a, vector_b):
    product = dot_product(
        vector_a,
        vector_b,
    )
    magnitude_a = vector_magnitude(vector_a)
    magnitude_b = vector_magnitude(vector_b)

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError(
            "zero vector has no direction"
        )

    return product / (
        magnitude_a * magnitude_b
    )


def retrieve_relevant_chunks(
    question_text,
    retrieval_index,
    top_k=3,
    min_score=0.0,
):
    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero"
        )

    if not 0.0 <= min_score <= 1.0:
        raise ValueError(
            "min_score must be between 0 and 1"
        )

    vocabulary = retrieval_index["vocabulary"]
    idf_weights = retrieval_index["idf_weights"]
    indexed_chunks = retrieval_index["chunks"]

    question_vector = text_to_tfidf_vector(
        question_text,
        vocabulary,
        idf_weights,
    )

    if vector_magnitude(question_vector) == 0:
        raise ValueError(
            "question does not contain indexed terms"
        )

    retrieval_results = []

    for indexed_chunk in indexed_chunks:
        similarity_score = cosine_similarity(
            question_vector,
            indexed_chunk["vector"],
        )

        if similarity_score >= min_score:
            retrieval_result = indexed_chunk.copy()
            retrieval_result["score"] = similarity_score

            retrieval_results.append(
                retrieval_result
            )

    retrieval_results.sort(
        key=lambda retrieval_result: (
            retrieval_result["score"]
        ),
        reverse=True,
    )

    return retrieval_results[:top_k]


if __name__ == "__main__":
    document_paths = [
        "sample_docs/python_venv.txt",
        "sample_docs/rag_notes.md",
    ]

    knowledge_chunks = []

    for document_path in document_paths:
        loaded_document = load_document(
            document_path
        )

        document_chunks = chunk_document(
            loaded_document,
            chunk_size=80,
            overlap=20,
        )

        knowledge_chunks.extend(
            document_chunks
        )

    retrieval_index = build_retrieval_index(
        knowledge_chunks
    )


    question_text = (
        "如何 隔离 Python 项目 依赖"
    )

    retrieval_results = retrieve_relevant_chunks(
        question_text,
        retrieval_index,
        top_k=2,
        min_score=0.1,
    )

    print("问题：", question_text)
    print(
        "检索结果数量：",
        len(retrieval_results),
    )

    for retrieval_result in retrieval_results:
        print("-" * 40)
        print(
            "相似度：",
            f'{retrieval_result["score"]:.4f}',
        )
        print(
            "来源：",
            retrieval_result["source"],
        )
        print(
            "路径：",
            retrieval_result["path"],
        )
        print(
            "切片编号：",
            retrieval_result["chunk_id"],
        )
        print(
            "字符范围：",
            retrieval_result["start"],
            "到",
            retrieval_result["end"],
        )
        print(
            "正文：",
            retrieval_result["text"],
        )
