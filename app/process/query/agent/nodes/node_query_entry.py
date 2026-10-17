from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_query_entry")
def node_query_entry(state):
    task_id = state.get("progress_id") or state.get("session_id", "")
    is_stream = state.get("is_stream", False)
    add_running_task(task_id, "node_query_entry", is_stream)
    add_done_task(task_id, "node_query_entry", is_stream)
    return {
        "original_query": state.get("original_query", ""),
        "rewritten_query": state.get("original_query", ""),
    }
