from pydantic import BaseModel, Field

from app.shared.config.audiobook_config import ContentType


class UploadResponse(BaseModel):
    code: int
    messages: str
    task_ids: list[str]


class StatusResponse(BaseModel):
    code: int
    status: str
    task_id: str
    done_list: list[str]
    running_list: list[str]
    metadata: dict = Field(default_factory=dict)


class AudiobookMetadata(BaseModel):
    content: str = ""
    content_type: ContentType
    book_name: str | None = Field(default=None, min_length=1, max_length=512)
    author_name: str | None = Field(default=None, max_length=256)
    item_name: str | None = Field(default=None, min_length=1, max_length=512)
    category: str | None = Field(default=None, max_length=128)
    duration: str | None = Field(default=None, max_length=64)
    source_file: str = Field(min_length=1, max_length=512)
    source_path: str | None = Field(default=None, max_length=1024)
    scene_tags: list[str] = Field(default_factory=list)