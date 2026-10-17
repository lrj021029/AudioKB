"""从原文档提取校验有声书原数据."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from app.infra.llm.providers import llm_provider
from app.shared.config.audiobook_config import (
    CATEGORIES,
    CONTENT_TYPES,
    METADATA_EXTRACT_MAX_CHARS,
    SCENE_TAGS,
)
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log


def _normalize_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [tag.strip() for tag in value.replace("，", ",").split(",")]
    elif isinstance(value, list):
        values = [str(tag).strip() for tag in value]
    else:
        values = []
    return list(dict.fromkeys(tag for tag in values if tag))


def _extract_with_llm(md_content: str, file_title: str) -> dict[str, Any]:
    prompt = load_prompt(
        "metadata_extract",
        file_title=file_title,
        md_content=md_content[:METADATA_EXTRACT_MAX_CHARS],
        content_types=json.dumps(CONTENT_TYPES, ensure_ascii=False),
        categories=json.dumps(CATEGORIES, ensure_ascii=False),
        scene_tags=json.dumps(SCENE_TAGS, ensure_ascii=False),
    )
    parser = JsonOutputParser()
    chain = llm_provider.chat(json_model=True) | parser
    result = chain.invoke([HumanMessage(content=prompt)])
    if not isinstance(result, dict):
        raise ValueError("metadata extraction response must be a JSON object")
    return result


@step_log("extract_audiobook_metadata")
def extract_metadata(
    md_content: str,
    file_title: str,
    provided_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not md_content.strip():
        raise ValueError("md_content cannot be empty")
    supplied = provided_metadata or {}
    extracted = _extract_with_llm(md_content, file_title)
    metadata = {**extracted, **{key: value for key, value in supplied.items() if value not in (None, "", [])}}

    content_type = str(metadata.get("content_type") or "有声书信息").strip()
    if content_type not in CONTENT_TYPES:
        raise ValueError(f"unsupported content_type: {content_type}")
    book_name = str(metadata.get("book_name") or file_title).strip()
    if not book_name:
        raise ValueError("book_name is required")

    category = str(metadata.get("category") or "").strip()
    if category and category not in CATEGORIES:
        logger.warning(f"未知类别 {category!r}，保留为自定义类别")
    tags = _normalize_tags(metadata.get("scene_tags"))
    unknown_tags = [tag for tag in tags if tag not in SCENE_TAGS]
    if unknown_tags:
        logger.warning(f"保留自定义场景标签：{unknown_tags}")

    return {
        "content_type": content_type,
        "book_name": book_name,
        "author_name": str(metadata.get("author_name") or "").strip(),
        "category": category,
        "scene_tags": tags,
        "duration": str(metadata.get("duration") or "").strip(),
    }


def extract_metadata_for_state(state: dict[str, Any]) -> dict[str, Any]:
    supplied = state.get("metadata") or {}
    metadata = extract_metadata(
        md_content=str(state.get("md_content") or ""),
        file_title=str(state.get("file_title") or ""),
        provided_metadata=state.get("metadata") or {},
    )
    state.update(metadata)
    state["metadata"] = metadata
    state["book_name"] = metadata["book_name"]
    state["item_name"] = str(
        state.get("item_name") or supplied.get("item_name") or metadata["book_name"]
    )
    state["source_file"] = str(
        state.get("source_file") or supplied.get("source_file") or state.get("file_title") or ""
    )
    state["source_path"] = str(
        state.get("source_path") or supplied.get("source_path") or state.get("local_file_path") or ""
    )
    for chunk in state.get("chunks", []):
        chunk.update({
            "content_type": metadata["content_type"],
            "book_name": metadata["book_name"],
            "author_name": metadata["author_name"],
            "category": metadata["category"],
            "scene_tags": ",".join(metadata["scene_tags"]),
            "duration": metadata["duration"],
            "source_file": state["source_file"],
            "source_path": state["source_path"],
        })
    return state
