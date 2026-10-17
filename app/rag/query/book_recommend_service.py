"""从向量集合中检索并分组有声书推荐。"""
from __future__ import annotations

from app.rag.query.embedding_search_service import search_domain_content
from app.rag.query.config import AUDIOBOOK_RECOMMEND_TOP_K


def recommend_books(query: str, filter_expr: str = "") -> list[dict]:
    docs = search_domain_content(query, filter_expr=filter_expr, top_k=AUDIOBOOK_RECOMMEND_TOP_K)
    books: dict[str, dict] = {}
    for doc in docs:
        name = doc.get("book_name") or doc.get("item_name") or ""
        if not name:
            continue
        book = books.setdefault(
            name,
            {
                "book_name": name,
                "author_name": doc.get("author_name", ""),
                "category": doc.get("category", ""),
                "scene_tags": doc.get("scene_tags", ""),
                "content_type": doc.get("content_type", ""),
                "features": [],
                "audience": "",
                "reason": "",
                "sources": [],
            },
        )
        if doc.get("content"):
            book["features"].append(doc["content"])
        book["sources"].append(doc)
    return list(books.values())
