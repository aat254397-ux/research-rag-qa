from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app import (
    answer_document_question,
    build_document_qa_index,
)


# 服务启动时准备索引，后续问题共用这份索引
@asynccontextmanager
async def lifespan(application: FastAPI):
    project_dir = Path(__file__).resolve().parent

    document_paths = [
        project_dir / "sample_docs" / "python_venv.txt",
        project_dir / "sample_docs" / "rag_notes.md",
    ]

    application.state.retrieval_index = build_document_qa_index(
        document_paths,
        chunk_size=80,
        overlap=20,
    )

    yield


api = FastAPI(
    title="Document QA API",
    version="0.2.0",
    lifespan=lifespan,
)


# 规定请求需要包含哪些字段，以及字段的限制
class QuestionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=2, ge=1, le=10)
    min_score: float = Field(default=0.1, ge=0.0, le=1.0)


@api.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "document-qa",
    }


@api.post("/ask")
def ask_question(request: QuestionRequest):
    # 当前使用模拟客户端，后面替换为真实模型客户端
    demo_client = Mock()
    demo_client.responses.create.return_value = SimpleNamespace(
        output_text="这是模拟回答，尚未调用真实模型。"
    )

    try:
        result = answer_document_question(
            question_text=request.question,
            retrieval_index=api.state.retrieval_index,
            client=demo_client,
            model="test-model",
            top_k=request.top_k,
            min_score=request.min_score,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    result["mode"] = "mock"

    return result
