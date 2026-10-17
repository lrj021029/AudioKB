import ast
import uuid
from mimetypes import guess_type
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import EventSourceResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.schema.query_schema import (
    BookDetailRequest,
    BookDetailResponse,
    BookRecommendRequest,
    BookRecommendResponse,
    HistoryDeleteResponse,
    HistoryItemResponse,
    HistoryResponse,
    QueryAsyncResponse,
    QueryRequest,
    QueryResponse,
)
from app.infra.persistence.history_repository import history_repository
from app.process.query.agent.main_graph import query_graph_app
from app.process.query.agent.state import QueryGraphState, create_query_default_state
from app.shared.runtime.logger import logger, PROJECT_ROOT
from app.shared.utils.sse_utils import (
    SSEEvent,
    create_sse_queue,
    push_to_session,
    sse_generator,
)
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    clear_task,
    get_done_task_list,
    update_task_status,
)

app = FastAPI(description="听书知识库查询模块")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                pass
        return [value]
    return [value]


@app.get("/html")
def return_chat_html():
    html_path = PROJECT_ROOT / "app" / "resources" / "html" / "chat.html"
    return FileResponse(path=str(html_path), media_type=guess_type(html_path)[0])


@app.get("/book-detail")
def return_book_detail_html():
    html_path = PROJECT_ROOT / "app" / "resources" / "html" / "book_detail.html"
    return FileResponse(path=str(html_path), media_type=guess_type(html_path)[0])


@app.get("/health")
def health():
    return {"code": 0}


@app.get("/stream/{session_id}")
def stream(session_id: str, request: Request):
    return EventSourceResponse(sse_generator(session_id, request))


def invoke_query_graph(
    session_id: str,
    original_query: str,
    is_stream: bool,
    requested_intent: str | None = None,
    filters: dict | None = None,
    web_search_enabled: bool = False,
    progress_id: str | None = None,
):
    progress_id = progress_id or session_id
    state: QueryGraphState = create_query_default_state(
        session_id=session_id,
        progress_id=progress_id,
        original_query=original_query,
        is_stream=is_stream,
        requested_intent=requested_intent or "",
        filters=filters or {},
        web_search_enabled=web_search_enabled,
    )
    try:
        update_task_status(progress_id, TASK_STATUS_PROCESSING, is_stream)
        result = query_graph_app.invoke(state)
        update_task_status(progress_id, TASK_STATUS_COMPLETED, is_stream)
        if is_stream:
            push_to_session(
                progress_id,
                SSEEvent.FINAL,
                {
                    "answer": result.get("answer"),
                    "status": TASK_STATUS_COMPLETED,
                    "image_urls": result.get("image_urls", []),
                    "citations": result.get("citations", []),
                    "intent": result.get("intent", ""),
                    "recommendations": result.get("recommendations", []),
                    "book_detail": result.get("book_detail", []),
                    "warnings": result.get("warnings", []),
                },
            )
        return result
    except Exception as exc:
        update_task_status(progress_id, TASK_STATUS_FAILED, is_stream)
        logger.exception(f"查询会话 {session_id} 失败")
        if is_stream:
            push_to_session(progress_id, SSEEvent.ERROR, {"error": str(exc)})
        raise


def _submit_query(
    tasks: BackgroundTasks,
    *,
    query: str,
    session_id: str | None,
    is_stream: bool,
    intent: str | None = None,
    filters: dict | None = None,
    web_search_enabled: bool = False,
):
    session_id = session_id or str(uuid.uuid4())
    progress_id = str(uuid.uuid4()) if is_stream else session_id
    clear_task(progress_id)
    if is_stream:
        create_sse_queue(progress_id)
        tasks.add_task(
            invoke_query_graph,
            session_id=session_id,
            original_query=query,
            is_stream=True,
            requested_intent=intent,
            filters=filters,
            web_search_enabled=web_search_enabled,
            progress_id=progress_id,
        )
        return QueryAsyncResponse(
            message=f"{session_id} 正在查询",
            session_id=session_id,
            stream_id=progress_id,
        )
    return invoke_query_graph(
        session_id=session_id,
        original_query=query,
        is_stream=False,
        requested_intent=intent,
        filters=filters,
        web_search_enabled=web_search_enabled,
        progress_id=progress_id,
    )


def _query_response(result: dict, session_id: str) -> QueryResponse:
    return QueryResponse(
        message=f"{session_id} 处理完毕",
        session_id=session_id,
        answer=result.get("answer", ""),
        done_list=get_done_task_list(session_id),
        image_urls=result.get("image_urls", []),
        citations=result.get("citations", []),
        intent=result.get("intent", ""),
        warnings=result.get("warnings", []),
    )


@app.post("/query")
def query(tasks: BackgroundTasks, params: QueryRequest):
    result = _submit_query(
        tasks,
        query=params.query,
        session_id=params.session_id,
        is_stream=params.is_stream,
        web_search_enabled=params.web_search_enabled,
    )
    if isinstance(result, QueryAsyncResponse):
        return result
    return _query_response(result, params.session_id)


@app.post("/query/chat")
def chat(tasks: BackgroundTasks, params: QueryRequest):
    return query(tasks, params)


@app.post("/query/recommend")
def recommend(tasks: BackgroundTasks, params: BookRecommendRequest):
    filters = {
        key: value
        for key, value in {
            "book_name": params.book_name,
            "category": params.category,
            "scene_tag": params.scene_tag,
            "author_name": params.author_name,
        }.items()
        if value
    }
    session_id = params.session_id or str(uuid.uuid4())
    result = _submit_query(
        tasks,
        query=params.query,
        session_id=session_id,
        is_stream=params.is_stream,
        intent="book_recommend",
        filters=filters,
        web_search_enabled=params.web_search_enabled,
    )
    if isinstance(result, QueryAsyncResponse):
        return result
    return BookRecommendResponse(
        session_id=session_id,
        answer=result.get("answer", ""),
        recommendations=result.get("recommendations", []),
        citations=result.get("citations", []),
        warnings=result.get("warnings", []),
    )


@app.post("/query/book_detail")
def book_detail(tasks: BackgroundTasks, params: BookDetailRequest):
    session_id = params.session_id or str(uuid.uuid4())
    result = _submit_query(
        tasks,
        query=params.book_name,
        session_id=session_id,
        is_stream=params.is_stream,
        intent="book_detail",
        filters={"book_name": params.book_name},
        web_search_enabled=params.web_search_enabled,
    )
    if isinstance(result, QueryAsyncResponse):
        return result
    return BookDetailResponse(
        session_id=session_id,
        book_name=params.book_name,
        answer=result.get("answer", ""),
        details=result.get("book_detail", []),
        citations=result.get("citations", []),
        warnings=result.get("warnings", []),
    )


@app.post("/query/search")
def search(tasks: BackgroundTasks, params: QueryRequest):
    result = _submit_query(
        tasks,
        query=params.query,
        session_id=params.session_id,
        is_stream=params.is_stream,
        intent="content_search",
        web_search_enabled=params.web_search_enabled,
    )
    if isinstance(result, QueryAsyncResponse):
        return result
    return _query_response(result, params.session_id)


@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    deleted_count = history_repository.clear_session(session_id=session_id)
    logger.info(f"共删除了:{deleted_count}条于{session_id}历史记录中")
    return HistoryDeleteResponse(
        message=f"共删除了:{deleted_count}条于{session_id}历史记录中",
        deleted_count=deleted_count,
    )


@app.get("/history/{session_id}")
@app.get("/query/history/{session_id}")
def list_history(session_id: str):
    history_message: list[dict] = history_repository.list_recent(session_id=session_id, limit=100)
    item_list: list[HistoryItemResponse] = [
        HistoryItemResponse(
            session_id=session_id,
            id=str(message.get("_id")),
            role=message.get("role", ""),
            text=message.get("text", ""),
            rewritten_query=message.get("rewritten_query", ""),
            item_names=_ensure_list(message.get("item_names")),
            image_urls=_ensure_list(message.get("image_urls")),
            ts=message.get("ts", 0.0),
            citations=message.get("citations", []),
            intent=message.get("intent", ""),
        )
        for message in history_message
    ]
    return HistoryResponse(session_id=session_id, items=item_list)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
