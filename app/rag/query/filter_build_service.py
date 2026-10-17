"""提取支持的搜索过滤器，并安全地构建 Milvus 表达式."""
from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from app.infra.llm.providers import llm_provider
from app.shared.config.audiobook_config import CATEGORIES, CONTENT_TYPES, SCENE_TAGS
from app.shared.runtime.load_prompt import load_prompt
from app.shared.utils.escape_milvus_string_utils import escape_milvus_string

_FILTER_FIELDS = {
    "book_name": "book_name",
    "author_name": "author_name",
    "category": "category",
    "scene_tag": "scene_tags",
    "content_type": "content_type",
}
_PARTIAL_MATCH_FIELDS = {"book_name", "author_name", "category"}


def extract_filters(query: str) -> dict[str, str]:
    if not query.strip():
        return {}
    parser = JsonOutputParser()
    result = (llm_provider.chat(json_model=True) | parser).invoke(
        [
            HumanMessage(
                content=load_prompt(
                    "filter_extract",
                    query=query,
                    content_types=", ".join(CONTENT_TYPES),
                )
            )
        ]
    )
    if not isinstance(result, dict):
        raise ValueError("过滤提取相应必须是一个JSON object")
    filters: dict[str, str] = {}
    for key in _FILTER_FIELDS:
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            normalized = value.strip()
            if key == "content_type" and normalized not in CONTENT_TYPES:
                continue
            filters[key] = normalized
    return filters


def build_filter_expression(
    filters: dict[str, str | list[str]] | None = None,
) -> str:
    expressions: list[str] = []
    for key, field in _FILTER_FIELDS.items():
        value = (filters or {}).get(key)
        if not value:
            continue
        if key == "content_type" and isinstance(value, list):
            values = [escape_milvus_string(item) for item in value if item]
            if values:
                escaped_values = ", ".join(f'"{item}"' for item in values)
                expressions.append(f"{field} in [{escaped_values}]")
            continue
        if not isinstance(value, str):
            continue
        escaped = escape_milvus_string(value)
        if key == "scene_tag" or key in _PARTIAL_MATCH_FIELDS:
            expressions.append(f'{field} like "%{escaped}%"')
        else:
            expressions.append(f'{field} == "{escaped}"')
    return " and ".join(expressions)


def build_filters_for_query(query: str) -> tuple[dict[str, str], str]:
    filters = extract_filters(query)
    # 保留明确的领域专有术语，即便大模型提取器省略或改写了这些术语。
    explicit_categories: list[str] = [str(value) for value in CATEGORIES if value in query]
    if explicit_categories:
        filters["category"] = sorted(explicit_categories, key=len, reverse=True)[0]
    explicit_scene_tags: list[str] = [str(value) for value in SCENE_TAGS if value in query]
    if explicit_scene_tags:
        filters["scene_tag"] = sorted(explicit_scene_tags, key=len, reverse=True)[0]
    return filters, build_filter_expression(filters)
