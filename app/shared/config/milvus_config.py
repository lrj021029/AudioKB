"""
Milvus 配置模块，负责读取向量库相关环境变量。
"""
from dataclasses import dataclass

from app.shared.config.common import env_str
from app.shared.config.audiobook_config import (
    MILVUS_AUTHOR_NAME_MAX_LENGTH,
    MILVUS_BOOK_NAME_MAX_LENGTH,
    MILVUS_CATEGORY_MAX_LENGTH,
    MILVUS_CONTENT_TYPE_MAX_LENGTH,
    MILVUS_DURATION_MAX_LENGTH,
    MILVUS_SCENE_TAGS_MAX_LENGTH,
    MILVUS_SOURCE_FILE_MAX_LENGTH,
    MILVUS_SOURCE_PATH_MAX_LENGTH,
)


@dataclass
class MilvusConfig:
    milvus_url: str
    chunks_collection: str
    entity_name_collection: str
    item_name_collection: str

milvus_config = MilvusConfig(
    milvus_url=env_str("MILVUS_URL"),
    chunks_collection=env_str("CHUNKS_COLLECTION"),
    entity_name_collection=env_str("ENTITY_NAME_COLLECTION"),
    item_name_collection=env_str("ITEM_NAME_COLLECTION"),
)

CHUNK_METADATA_FIELDS = {
    "content_type": MILVUS_CONTENT_TYPE_MAX_LENGTH,
    "book_name": MILVUS_BOOK_NAME_MAX_LENGTH,
    "author_name": MILVUS_AUTHOR_NAME_MAX_LENGTH,
    "category": MILVUS_CATEGORY_MAX_LENGTH,
    "scene_tags": MILVUS_SCENE_TAGS_MAX_LENGTH, # 场景标签
    "duration": MILVUS_DURATION_MAX_LENGTH, # 时长
    "source_file": MILVUS_SOURCE_FILE_MAX_LENGTH,
    "source_path": MILVUS_SOURCE_PATH_MAX_LENGTH,
}
