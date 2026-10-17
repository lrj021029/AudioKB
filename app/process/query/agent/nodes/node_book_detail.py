from app.rag.query.filter_build_service import build_filter_expression
from app.shared.config.audiobook_config import ContentType
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_book_detail")
def node_book_detail(state):
    """Mark a book-detail query for the shared retrieval and answer pipeline."""
    task_id = state.get("progress_id") or state.get("session_id", "")
    is_stream = state.get("is_stream", False)
    add_running_task(task_id, "node_book_detail", is_stream)
    filters = dict(state.get("filters") or {})
    book_name = filters.get("book_name") or state.get("book_name") or ""
    if book_name:
        filters["book_name"] = book_name
    target_content_types = [
        ContentType.BOOK_INTRODUCTION.value,
        ContentType.AUTHOR_INTRODUCTION.value,
        ContentType.AUDIOBOOK_INFO.value,
        ContentType.FAQ.value,
    ]
    filters["content_type"] = target_content_types
    filter_expr = build_filter_expression(filters)
    rerank_query = (
        f"{book_name} 的简介、作者、时长、演播信息、内容标签、听书亮点"
        if book_name
        else state.get("original_query", "")
    )
    add_done_task(task_id, "node_book_detail", is_stream)
    return {
        "filters": filters,
        "filter_expr": filter_expr,
        "target_content_types": target_content_types,
        "prompt_template": "book_detail",
        "rerank_query": rerank_query,
    }
