from app.rag.query.filter_build_service import build_filter_expression, build_filters_for_query
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_filter_build")
def node_filter_build(state):
    task_id = state.get("progress_id") or state.get("session_id", "")
    add_running_task(task_id, "node_filter_build", state.get("is_stream", False))
    filters, expression = build_filters_for_query(state.get("original_query", ""))
    filters.update({
        key: value for key, value in (state.get("filters") or {}).items()
        if value not in (None, "")
    })
    expression = build_filter_expression(filters)
    state["filters"] = filters
    state["filter_expr"] = expression
    add_done_task(task_id, "node_filter_build", state.get("is_stream", False))
    return {"filters": filters, "filter_expr": expression}
