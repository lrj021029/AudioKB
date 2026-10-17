from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger
from pathlib import Path
import json

def resolve_state(state: ImportGraphState) -> ImportGraphState:
    """
    校验 local_file_path
    识别文件类型
    回写 is_pdf/md_read_enabled

    :param state:
    :return:
    """
    local_file_path:str = state.get("local_file_path")
    # local_file_path 判断
    if not local_file_path:
        logger.error(f"local_file_path is null!")
        raise ValueError(f"local_file_path is null!")
    suffix = Path(local_file_path).suffix.lower()
    if suffix in {".md", ".txt", ".json"}:
        state["is_md_read_enabled"] = True
        state["is_pdf_read_enabled"] = False
        state["md_path"] = local_file_path
        if suffix != ".md" and not state.get("md_content"):
            raw_content = Path(local_file_path).read_text(encoding="utf-8-sig")
            if suffix == ".json":
                raw_content = json.dumps(json.loads(raw_content), ensure_ascii=False, indent=2)
            state["md_content"] = raw_content
    elif suffix == ".pdf":
        state["is_md_read_enabled"] = False
        state["is_pdf_read_enabled"] = True
        state["pdf_path"] = local_file_path
    else:
        return state

    # 地址转换为 Path 对象快速获取文件 file_title
    local_file_obj = Path(local_file_path)
    file_title:str = local_file_obj.stem
    state["file_title"] = file_title
    if not state.get("source_file"):
        state["source_file"] = local_file_obj.name
    if not state.get("source_path"):
        state["source_path"] = str(local_file_obj)

    return state