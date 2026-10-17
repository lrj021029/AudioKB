from app.rag.query.intent_service import classify_intent
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_intent_classify")
def node_intent_classify(state):
    task_id = state.get("progress_id") or state.get("session_id", "")
    add_running_task(task_id, "node_intent_classify", state.get("is_stream", False))
    state["intent"] = state.get("requested_intent") or classify_intent(state.get("original_query", ""))
    state["rewritten_query"] = state.get("original_query", "")
    add_done_task(task_id, "node_intent_classify", state.get("is_stream", False))
    return {"intent": state["intent"], "rewritten_query": state["rewritten_query"]}
