import shutil
import uuid
from datetime import datetime
from mimetypes import guess_type
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import EventSourceResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from app.api.schema.import_schema import AudiobookMetadata, StatusResponse, UploadResponse
from app.infra.persistence.import_task_repository import import_task_repository
from app.process.import_.agent.main_graph import graph_app
from app.process.import_.agent.state import ImportGraphState, create_default_state
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.shared.config.audiobook_config import ContentType
from app.shared.utils.sse_utils import SSEEvent, create_sse_queue, sse_generator
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PENDING,
    TASK_STATUS_PROCESSING,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
    update_task_status,
)
from fastapi.requests import Request

app = FastAPI(description="听书知识库导入模块")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
_ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt", ".json"}


def invoke_import_graph(
    task_id: str,
    local_file_path: str,
    local_dir: str,
    metadata: dict | None = None,
    is_stream: bool = False,
) -> None:
    state: ImportGraphState = create_default_state(
        task_id=task_id,
        local_file_path=local_file_path,
        local_dir=local_dir,
        metadata=metadata or {},
        is_stream=is_stream,
        source_file=(metadata or {}).get("source_file", ""),
        source_path=(metadata or {}).get("source_path", ""),
        item_name=(metadata or {}).get("item_name", ""),
        book_name=(metadata or {}).get("book_name", ""),
        content_type=(metadata or {}).get("content_type", ""),
        author_name=(metadata or {}).get("author_name", ""),
        category=(metadata or {}).get("category", ""),
        scene_tags=(metadata or {}).get("scene_tags", []),
        duration=(metadata or {}).get("duration", ""),
    )
    try:
        update_task_status(task_id, TASK_STATUS_PROCESSING, is_stream)
        import_task_repository.save(
            task_id,
            status=TASK_STATUS_PROCESSING,
            source_file=state.get("source_file", ""),
            metadata=metadata,
        )
        result_state = graph_app.invoke(state)
        resolved_metadata = {
            **(metadata or {}),
            **{
                key: result_state.get(key)
                for key in (
                    "content_type",
                    "book_name",
                    "author_name",
                    "category",
                    "scene_tags",
                    "duration",
                    "item_name",
                    "source_file",
                    "source_path",
                )
                if result_state.get(key) not in (None, "", [])
            },
        }
        update_task_status(task_id, TASK_STATUS_COMPLETED, is_stream)
        import_task_repository.save(
            task_id,
            status=TASK_STATUS_COMPLETED,
            source_file=resolved_metadata.get("source_file", state.get("source_file", "")),
            metadata=resolved_metadata,
        )
        if is_stream:
            from app.shared.utils.sse_utils import push_to_session

            push_to_session(
                task_id,
                SSEEvent.FINAL,
                {"status": TASK_STATUS_COMPLETED, "metadata": resolved_metadata},
            )
    except Exception as exc:
        logger.exception(f"导入任务 {task_id} 失败")
        update_task_status(task_id, TASK_STATUS_FAILED, is_stream)
        import_task_repository.save(
            task_id,
            status=TASK_STATUS_FAILED,
            source_file=state.get("source_file", ""),
            metadata=metadata,
            error=str(exc),
        )
        if is_stream:
            from app.shared.utils.sse_utils import push_to_session

            push_to_session(task_id, SSEEvent.ERROR, {"error": str(exc)})
        raise


def _save_upload(file: UploadFile, task_id: str) -> tuple[Path, Path]:
    filename = Path(file.filename or "upload.md").name
    if Path(filename).suffix.lower() not in _ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="仅支持 PDF、MD、TXT、JSON 文件")
    local_dir = PROJECT_ROOT / "output" / datetime.now().strftime("%Y%m%d") / task_id
    local_dir.mkdir(parents=True, exist_ok=True)
    target = local_dir / filename
    with target.open("wb") as stream:
        shutil.copyfileobj(file.file, stream)
    return target, local_dir


async def _schedule_domain_import(
    tasks: BackgroundTasks,
    *,
    file: UploadFile | None,
    content: str | None,
    content_type: str,
    book_name: str | None,
    item_name: str | None,
    author_name: str | None,
    category: str | None,
    scene_tags: str | None,
    duration: str | None,
    source_file: str | None,
    source_path: str | None,
    default_content_type: str,
) -> UploadResponse:
    if file is None and not (content and content.strip()):
        raise HTTPException(status_code=422, detail="请上传文件或填写内容正文")
    task_id = str(uuid.uuid4())
    if file is not None:
        local_path, local_dir = _save_upload(file, task_id)
        source_name = (source_file or Path(file.filename or local_path.name).name)[:512]
    else:
        local_dir = PROJECT_ROOT / "output" / datetime.now().strftime("%Y%m%d") / task_id
        local_dir.mkdir(parents=True, exist_ok=True)
        source_name = (source_file or f"{book_name or 'audiobook'}.md")[:512]
        local_path = local_dir / f"{Path(source_name).stem or 'audiobook'}.md"
        local_path.write_text(content or "", encoding="utf-8")
    try:
        metadata_model = AudiobookMetadata(
            content=content or "",
            content_type=ContentType(content_type or default_content_type),
            book_name=(book_name.strip() or None) if book_name else None,
            item_name=(item_name.strip() or None) if item_name else None,
            author_name=author_name or None,
            category=category or None,
            duration=duration or None,
            source_file=source_name,
            source_path=source_path or str(local_path),
            scene_tags=[
                tag.strip()
                for tag in (scene_tags or "").replace("，", ",").split(",")
                if tag.strip()
            ],
        )
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    metadata = metadata_model.model_dump(
        mode="json",
        exclude={"content", "source_file", "source_path"},
        exclude_none=True,
    )
    metadata.update(
        source_file=metadata_model.source_file,
        source_path=metadata_model.source_path,
    )
    create_sse_queue(task_id)
    update_task_status(task_id, TASK_STATUS_PENDING, True)
    import_task_repository.save(
        task_id,
        status=TASK_STATUS_PENDING,
        source_file=metadata_model.source_file,
        metadata=metadata,
    )
    tasks.add_task(
        invoke_import_graph,
        task_id=task_id,
        local_file_path=str(local_path),
        local_dir=str(local_dir),
        metadata=metadata,
        is_stream=True,
    )
    return UploadResponse(
        code=200,
        messages=f"{metadata_model.source_file} 已接收，正在导入",
        task_ids=[task_id],
    )


@app.get("/html")
def return_import_html():
    html_file_obj = PROJECT_ROOT / "app" / "resources" / "html" / "import.html"
    return FileResponse(path=str(html_file_obj), media_type=guess_type(html_file_obj)[0])


@app.get("/stream/{task_id}")
def import_stream(task_id: str, request: Request):
    return EventSourceResponse(sse_generator(task_id, request))


@app.post("/upload")
async def upload_file(tasks: BackgroundTasks, files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=422, detail="至少选择一个文件")
    responses: list[str] = []
    for file in files:
        task_id = str(uuid.uuid4())
        local_path, local_dir = _save_upload(file, task_id)
        update_task_status(task_id, TASK_STATUS_PENDING)
        import_task_repository.save(
            task_id,
            status=TASK_STATUS_PENDING,
            source_file=local_path.name,
            metadata={},
        )
        tasks.add_task(
            invoke_import_graph,
            task_id=task_id,
            local_file_path=str(local_path),
            local_dir=str(local_dir),
        )
        responses.append(task_id)
    return UploadResponse(
        code=200,
        messages=f"已接收 {len(responses)} 个文件，正在解析",
        task_ids=responses,
    )


@app.post("/import/audiobook")
async def import_audiobook(
    tasks: BackgroundTasks,
    file: UploadFile | None = File(None),
    content: str | None = Form(None),
    content_type: str = Form("有声书信息"),
    book_name: str | None = Form(None),
    item_name: str | None = Form(None),
    author_name: str | None = Form(None),
    category: str | None = Form(None),
    scene_tags: str | None = Form(None),
    duration: str | None = Form(None),
    source_file: str | None = Form(None),
    source_path: str | None = Form(None),
):
    return await _schedule_domain_import(
        tasks,
        file=file,
        content=content,
        content_type=content_type,
        book_name=book_name,
        item_name=item_name,
        author_name=author_name,
        category=category,
        scene_tags=scene_tags,
        duration=duration,
        source_file=source_file,
        source_path=source_path,
        default_content_type="有声书信息",
    )


@app.post("/import/note")
async def import_note(
    tasks: BackgroundTasks,
    file: UploadFile | None = File(None),
    content: str | None = Form(None),
    book_name: str | None = Form(None),
    item_name: str | None = Form(None),
    author_name: str | None = Form(None),
    category: str | None = Form(None),
    scene_tags: str | None = Form(None),
    duration: str | None = Form(None),
    source_file: str | None = Form(None),
    source_path: str | None = Form(None),
):
    return await _schedule_domain_import(
        tasks,
        file=file,
        content=content,
        content_type="听书笔记",
        book_name=book_name,
        item_name=item_name,
        author_name=author_name,
        category=category,
        scene_tags=scene_tags,
        duration=duration,
        source_file=source_file,
        source_path=source_path,
        default_content_type="听书笔记",
    )


@app.post("/import/qa")
async def import_qa(
    tasks: BackgroundTasks,
    file: UploadFile | None = File(None),
    content: str | None = Form(None),
    book_name: str | None = Form(None),
    item_name: str | None = Form(None),
    author_name: str | None = Form(None),
    category: str | None = Form(None),
    scene_tags: str | None = Form(None),
    duration: str | None = Form(None),
    source_file: str | None = Form(None),
    source_path: str | None = Form(None),
):
    return await _schedule_domain_import(
        tasks,
        file=file,
        content=content,
        content_type="常见问答",
        book_name=book_name,
        item_name=item_name,
        author_name=author_name,
        category=category,
        scene_tags=scene_tags,
        duration=duration,
        source_file=source_file,
        source_path=source_path,
        default_content_type="常见问答",
    )


@app.get("/status/{task_id}")
def status_info(task_id: str):
    record = import_task_repository.get(task_id)
    status = get_task_status(task_id) or (record or {}).get("status", "")
    return StatusResponse(
        code=200,
        task_id=task_id,
        status=status,
        running_list=get_running_task_list(task_id),
        done_list=get_done_task_list(task_id),
        metadata=(record or {}).get("metadata", {}),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
