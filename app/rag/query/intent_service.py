"""有声书查询意图分类"""
from __future__ import annotations

from enum import StrEnum
import re

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from app.infra.llm.providers import llm_provider
from app.shared.runtime.load_prompt import load_prompt


class QueryIntent(StrEnum):
    BOOK_RECOMMEND = "book_recommend"
    BOOK_DETAIL = "book_detail"
    CONTENT_SEARCH = "content_search"
    NOTE_QUERY = "note_query"
    BOOK_COUNT = "book_count"
    GENERAL_QA = "general_qa"


_BOOK_COUNT_QUERY = re.compile(
    r"(?:知识库|本地库|馆藏).{0,12}(?:有多少|多少|几本|几部|几册).{0,8}"
    r"(?:有声书|书籍|本书|书|藏书)|"
    r"(?:一共|总共|共有|收录|包含).{0,8}(?:有多少|多少|几本|几部|几册).{0,8}"
    r"(?:有声书|书籍|本书|书|藏书)|"
    r"(?:有多少|多少本|多少部|多少册).{0,8}"
    r"(?:有声书|书籍|本书|书|藏书)"
)


def classify_intent(query: str) -> str:
    if not query.strip():
        raise ValueError("query cannot be empty")
    if _BOOK_COUNT_QUERY.search(query):
        return QueryIntent.BOOK_COUNT.value
    parser = JsonOutputParser()
    result = (llm_provider.chat(json_model=True) | parser).invoke(
        [HumanMessage(content=load_prompt("intent_classify", query=query))]
    )
    intent = str(result.get("intent", "")).strip() if isinstance(result, dict) else ""
    return intent if intent in {item.value for item in QueryIntent} else QueryIntent.GENERAL_QA.value
