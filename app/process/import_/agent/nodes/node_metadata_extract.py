from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.metadata_extract_service import extract_metadata_for_state
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_metadata_extract")
def node_metadata_extract(state: ImportGraphState) -> ImportGraphState:
    task_id = state.get("task_id", "")
    add_running_task(task_id, "node_metadata_extract", state.get("is_stream", False))
    state = extract_metadata_for_state(state)
    add_done_task(task_id, "node_metadata_extract", state.get("is_stream", False))
    return state
