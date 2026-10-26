from contextlib import asynccontextmanager
import logging
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Literal
from unittest.mock import Mock

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from openai import APIError, APITimeoutError
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pypdf.errors import PdfReadError

from app import (
    answer_document_question,
    build_document_qa_index,
    search_document_references,
)
from deepseek_client import create_deepseek_client, model_service_error_detail
from embedding_retrieval import (
    DIMENSION,
    MODEL_NAME,
    MODEL_REVISION,
    E5Encoder,
    EmbeddingUnavailableError,
    build_embedding_index,
    corpus_fingerprint,
)
from embedding_store import (
    load_embedding_index,
    save_embedding_index,
    validate_embedding_index,
)
from index_store import load_retrieval_index, save_retrieval_index
from retrieval import retrieve_relevant_chunks
from reranking import (
    DEFAULT_BATCH_SIZE as RERANK_BATCH_SIZE,
    MAX_PAIR_TOKENS,
    MODEL_NAME as RERANK_MODEL_NAME,
    MODEL_REVISION as RERANK_MODEL_REVISION,
    CrossEncoderReranker,
    RerankerUnavailableError,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INDEX_PATH = PROJECT_DIR / "data" / "indexes" / "retrieval_index.json"
WEB_DIR = PROJECT_DIR / "web" / "dist"
LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    snapshot_path = application.state.index_path
    if snapshot_path is not None and snapshot_path.exists():
        try:
            retrieval_index = load_retrieval_index(snapshot_path)
        except (OSError, ValueError) as error:
            # 不悄悄退回示例库，避免用户误以为仍在查询自己的论文。
            raise RuntimeError(
                f"无法恢复索引快照：{snapshot_path}。"
                "请先备份并移走该文件，再启动服务、重新加载资料。"
            ) from error
        index_origin = "disk"
    else:
        document_paths = [
            PROJECT_DIR / "sample_docs" / "python_venv.txt",
            PROJECT_DIR / "sample_docs" / "rag_notes.md",
        ]
        retrieval_index = build_document_qa_index(document_paths)
        index_origin = "sample_documents"

    application.state.retrieval_index = retrieval_index
    application.state.index_origin = index_origin
    # 单进程内，避免两次加载交错保存，导致磁盘和内存不是同一版。
    application.state.index_update_lock = Lock()
    application.state.corpus_fingerprint = corpus_fingerprint(retrieval_index["chunks"])
    application.state.embedding_encoder = E5Encoder()
    application.state.reranker = CrossEncoderReranker()
    restore_embedding_snapshot(application)

    yield


class SearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=2, ge=1, le=10)
    retriever: Literal["tfidf", "embedding"] = "tfidf"
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    rerank: bool = Field(default=False, description="是否对 E5 候选进行重排")
    candidate_k: int = Field(
        default=10, ge=1, le=50,
        description="开启重排时的候选数量上限，必须不小于 top_k",
    )

    @model_validator(mode="after")
    def validate_retriever_threshold(self):
        if self.retriever == "tfidf" and self.min_score is not None:
            if self.min_score < 0:
                raise ValueError("TF-IDF min_score must be between 0 and 1")
        if self.rerank:
            if self.retriever != "embedding":
                raise ValueError("本版重排请同时设置 retriever=embedding")
            if self.candidate_k < self.top_k:
                raise ValueError("candidate_k 必须大于或等于 top_k")
        return self


class QuestionRequest(SearchRequest):
    # 问答继承检索参数，另加模型模式；纯检索不需要 mode。
    mode: Literal["mock", "deepseek"] = "mock"


class DocumentLoadRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    paths: list[str] = Field(min_length=1)


def home():
    # 只提供前端目录，不把项目根目录、.env 或文献快照暴露为静态文件。
    return FileResponse(
        WEB_DIR / "index.html", media_type="text/html",
        headers={"Cache-Control": "no-cache"},
    )


def favicon():
    # 兼容浏览器自动访问的旧路径，实际图标仍由静态目录提供。
    return RedirectResponse(url="/assets/favicon.svg", status_code=307)


def health_check():
    return {
        "status": "ok",
        "service": "document-qa",
    }


def describe_index(retrieval_index):
    chunks = retrieval_index["chunks"]
    document_paths = sorted({chunk["path"] for chunk in chunks})
    page_keys = {
        (chunk["path"], chunk["page"])
        for chunk in chunks
        if "page" in chunk
    }
    return {
        "document_count": len(document_paths),
        "indexed_pdf_pages": len(page_keys),
        "chunk_count": len(chunks),
        "documents": document_paths,
    }


def describe_snapshot(snapshot_path):
    size_bytes = 0
    exists = False
    if snapshot_path is not None:
        try:
            size_bytes = snapshot_path.stat().st_size
            exists = True
        except FileNotFoundError:
            pass
    return {
        "snapshot_exists": exists,
        "snapshot_path": str(snapshot_path) if snapshot_path else None,
        "snapshot_size_bytes": size_bytes,
    }


def restore_embedding_snapshot(application):
    """只读取向量快照，不加载 E5 权重，也不重新编码全文。"""
    application.state.embedding_index = None
    application.state.embedding_status = "not_built"
    application.state.embedding_origin = None
    application.state.embedding_error = None
    path = application.state.embedding_index_path
    if path is None or not path.exists():
        return
    try:
        index = load_embedding_index(path)
    except (OSError, ValueError) as error:
        application.state.embedding_status = "invalid"
        application.state.embedding_error = "E5 快照损坏或编码配置不兼容，请重新构建。"
        LOGGER.warning("无法恢复 E5 快照 %s: %s", path, error)
        return
    if index["source_fingerprint"] != application.state.corpus_fingerprint:
        application.state.embedding_status = "stale"
        application.state.embedding_error = "资料版本已变化，请为当前资料重新构建 E5 索引。"
        return
    application.state.embedding_index = index
    application.state.embedding_status = "ready"
    application.state.embedding_origin = "disk"


def index_status(http_request: Request):
    application = http_request.app
    with application.state.index_update_lock:
        snapshot_path = application.state.index_path
        snapshot = describe_snapshot(snapshot_path)
        embedding_index = application.state.embedding_index

        return {
            **describe_index(application.state.retrieval_index),
            "index_origin": application.state.index_origin,
            "index_storage": "memory+disk" if snapshot["snapshot_exists"] else "memory",
            "persistence_enabled": snapshot_path is not None,
            **snapshot,
            "embedding": {
                "status": application.state.embedding_status,
                "origin": application.state.embedding_origin,
                "model_name": MODEL_NAME,
                "model_revision": MODEL_REVISION,
                "dimension": DIMENSION,
                "chunk_count": len(embedding_index["chunks"]) if embedding_index else 0,
                "model_loaded": application.state.embedding_encoder.loaded,
                "detail": application.state.embedding_error,
                **describe_snapshot(application.state.embedding_index_path),
            },
            "reranker": {
                "model_name": RERANK_MODEL_NAME,
                "model_revision": RERANK_MODEL_REVISION,
                "model_loaded": application.state.reranker.loaded,
                "device": "cpu",
                "batch_size": RERANK_BATCH_SIZE,
                "max_pair_tokens": MAX_PAIR_TOKENS,
            },
        }


def load_documents(request: DocumentLoadRequest, http_request: Request):
    document_paths = []

    for path_text in request.paths:
        if not path_text:
            raise HTTPException(
                status_code=400,
                detail="资料路径不能为空。",
            )
        document_paths.append(Path(path_text).resolve())

    try:
        new_index = build_document_qa_index(document_paths)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    except (PdfReadError, OSError) as error:
        raise HTTPException(
            status_code=400,
            detail="无法读取资料，请检查文件权限及 PDF 是否完整。",
        ) from error

    application = http_request.app
    snapshot_path = application.state.index_path
    with application.state.index_update_lock:
        if snapshot_path is not None:
            try:
                save_retrieval_index(new_index, snapshot_path)
            except (OSError, ValueError, TypeError) as error:
                raise HTTPException(
                    status_code=500,
                    detail="索引保存失败，本次未切换索引，请检查磁盘空间和目录权限。",
                ) from error

        # 建立和保存均成功后，再切换内存索引；失败时保留旧版。
        application.state.retrieval_index = new_index
        application.state.index_origin = "loaded_documents"
        application.state.corpus_fingerprint = corpus_fingerprint(new_index["chunks"])
        restore_embedding_snapshot(application)

    return {
        "status": "loaded",
        **describe_index(new_index),
        "index_storage": "memory+disk" if snapshot_path else "memory",
    }


def build_e5_index(http_request: Request):
    application = http_request.app
    # 仅取当前索引的稳定引用；编码期间允许旧索引继续提供检索。
    with application.state.index_update_lock:
        source_index = application.state.retrieval_index
        source_fingerprint = application.state.corpus_fingerprint

    try:
        new_index = build_embedding_index(
            source_index["chunks"], application.state.embedding_encoder
        )
        validate_embedding_index(new_index)
    except EmbeddingUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    with application.state.index_update_lock:
        if application.state.corpus_fingerprint != source_fingerprint:
            raise HTTPException(
                status_code=409,
                detail="编码期间资料库发生变化，本次未保存索引，请重新构建。",
            )
        path = application.state.embedding_index_path
        if path is not None:
            try:
                save_embedding_index(new_index, path)
            except (OSError, ValueError, TypeError) as error:
                raise HTTPException(
                    status_code=500,
                    detail="E5 索引保存失败，本次未切换，请检查磁盘空间和目录权限。",
                ) from error
        application.state.embedding_index = new_index
        application.state.embedding_status = "ready"
        application.state.embedding_origin = "built"
        application.state.embedding_error = None

    return {
        "status": "ready",
        "retriever": "embedding",
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "dimension": DIMENSION,
        **describe_index(new_index),
        "index_storage": "memory+disk" if path else "memory",
    }


def effective_min_score(request: SearchRequest):
    if request.min_score is not None:
        return request.min_score
    # 保持原 TF-IDF 行为；E5 首轮先不过滤，观察完整排名后再校准阈值。
    return 0.1 if request.retriever == "tfidf" else -1.0


def select_retrieval(request: SearchRequest, application: FastAPI):
    with application.state.index_update_lock:
        if request.retriever == "tfidf":
            return application.state.retrieval_index, retrieve_relevant_chunks
        index = application.state.embedding_index
        if index is None or index["source_fingerprint"] != application.state.corpus_fingerprint:
            raise HTTPException(
                status_code=409,
                detail="当前资料没有可用的 E5 索引，请先执行 POST /index/embedding/build。",
            )
        return index, application.state.embedding_encoder.retrieve


def search_documents(request: SearchRequest, http_request: Request):
    index, retrieval_function = select_retrieval(request, http_request.app)
    min_score = effective_min_score(request)
    try:
        references = search_document_references(
            question_text=request.question,
            retrieval_index=index,
            top_k=request.top_k,
            min_score=min_score,
            retrieval_function=retrieval_function,
            candidate_k=request.candidate_k,
            reranker=http_request.app.state.reranker if request.rerank else None,
        )
    except (EmbeddingUnavailableError, RerankerUnavailableError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    # 排名从 1 开始；不返回体积较大的完整向量。
    return {
        "question": request.question,
        "retriever": request.retriever,
        "effective_min_score": min_score,
        "rerank": request.rerank,
        "candidate_k": request.candidate_k if request.rerank else None,
        "match_count": len(references),
        "references": [
            {"rank": rank, **reference}
            for rank, reference in enumerate(references, start=1)
        ],
    }


def ask_question(request: QuestionRequest, http_request: Request):
    index, retrieval_function = select_retrieval(request, http_request.app)
    min_score = effective_min_score(request)
    # 根据请求选择客户端
    if request.mode == "deepseek":
        try:
            client, model = create_deepseek_client()
        except ValueError as error:
            raise HTTPException(
                status_code=503,
                detail="DeepSeek 配置不完整，请检查本地密钥、模型名和 API 地址。",
            ) from error
    else:
        client = Mock()
        client.responses.create.return_value = SimpleNamespace(
            output_text="这是模拟回答，尚未调用真实模型。"
        )
        model = "test-model"

    # 两种客户端共用现有问答流程
    try:
        result = answer_document_question(
            question_text=request.question,
            retrieval_index=index,
            client=client,
            model=model,
            top_k=request.top_k,
            min_score=min_score,
            retrieval_function=retrieval_function,
            candidate_k=request.candidate_k,
            reranker=http_request.app.state.reranker if request.rerank else None,
        )
    except (EmbeddingUnavailableError, RerankerUnavailableError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except APITimeoutError as error:
        raise HTTPException(
            status_code=504,
            detail="等待模型服务响应超时，请稍后重试。",
        ) from error
    except APIError as error:
        detail = model_service_error_detail(error)
        LOGGER.warning("%s", detail)
        raise HTTPException(
            status_code=502,
            detail=detail,
        ) from error
    except ValueError as error:
        # 暂时沿用现有问答函数的 ValueError 映射
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    finally:
        if request.mode == "deepseek":
            client.close()

    result["mode"] = request.mode
    result["retriever"] = request.retriever
    result["effective_min_score"] = min_score
    result["rerank"] = request.rerank
    result["candidate_k"] = request.candidate_k if request.rerank else None

    return result


def create_app(index_path=DEFAULT_INDEX_PATH):
    """正式服务使用磁盘快照；测试可传 None 或临时目录中的路径。"""
    application = FastAPI(
        title="Document QA API",
        version="0.9.0",
        lifespan=lifespan,
    )
    application.state.index_path = (
        Path(index_path).resolve() if index_path is not None else None
    )
    application.state.embedding_index_path = (
        application.state.index_path.with_name(
            application.state.index_path.stem + ".e5.json"
        )
        if application.state.index_path is not None else None
    )
    application.add_api_route("/", home, methods=["GET"], include_in_schema=False)
    application.add_api_route("/favicon.ico", favicon, methods=["GET"], include_in_schema=False)
    application.mount("/assets", StaticFiles(directory=str(WEB_DIR)), name="assets")
    application.add_api_route("/health", health_check, methods=["GET"])
    application.add_api_route("/index/status", index_status, methods=["GET"])
    application.add_api_route("/index/embedding/build", build_e5_index, methods=["POST"])
    application.add_api_route("/documents/load", load_documents, methods=["POST"])
    application.add_api_route("/search", search_documents, methods=["POST"])
    application.add_api_route("/ask", ask_question, methods=["POST"])
    return application


api = create_app()
