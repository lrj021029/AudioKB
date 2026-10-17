"""检索有声书详情及其源文本段落。"""
from __future__ import annotations

from app.rag.query.config import AUDIOBOOK_DETAIL_TOP_K
from app.rag.query.embedding_search_service import search_domain_content


def get_book_detail(book_name: str, filter_expr: str = "") -> list[dict]:
    if not book_name.strip():
        raise ValueError("book_name是必须的")
    return search_domain_content(
        book_name,
        filter_expr=filter_expr,
        top_k=AUDIOBOOK_DETAIL_TOP_K,
    )
