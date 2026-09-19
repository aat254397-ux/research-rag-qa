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

    print(
        "知识切片数量：",
        len(retrieval_index["chunks"]),
    )
    print(
        "词表大小：",
        len(retrieval_index["vocabulary"]),
    )
    print(
        "IDF权重数量：",
        len(retrieval_index["idf_weights"]),
    )
    print(
        "词表：",
        retrieval_index["vocabulary"],
    )

    for indexed_chunk in retrieval_index["chunks"]:
        print("-" * 40)
        print(
            "来源：",
            indexed_chunk["source"],
        )
        print(
            "切片编号：",
            indexed_chunk["chunk_id"],
        )
        print(
            "字符范围：",
            indexed_chunk["start"],
            "到",
            indexed_chunk["end"],
        )
        print(
            "向量：",
            indexed_chunk["vector"],
        )
        print(
            "正文：",
            indexed_chunk["text"],
        )
