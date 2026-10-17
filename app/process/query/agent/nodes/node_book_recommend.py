"""推荐书"""
from app.rag.query.filter_build_service import build_filter_expression
from app.shared.config.audiobook_config import ContentType
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_book_recommend")
def node_book_recommend(state):
    """Mark a recommendation query for the shared retrieval and answer pipeline."""
    task_id = state.get("progress_id") or state.get("session_id", "")
    is_stream = state.get("is_stream", False)
    add_running_task(task_id, "node_book_recommend", is_stream)
    filters = dict(state.get("filters") or {})
    target_content_types = [
        ContentType.AUDIOBOOK_INFO.value,
        ContentType.BOOK_INTRODUCTION.value,
        ContentType.LISTENING_NOTES.value,
        ContentType.USER_REVIEW_SUMMARY.value,
        ContentType.EDITORIAL_RECOMMENDATION.value,
    ]
    filters["content_type"] = target_content_types
    add_done_task(task_id, "node_book_recommend", is_stream)
    return {
        "filters": filters,
        "filter_expr": build_filter_expression(filters),
        "target_content_types": target_content_types,
        "prompt_template": "book_recommend",
        "rerank_query": state.get("rewritten_query") or state.get("original_query", ""),
    }
