"""知识库目录统计。"""
from app.infra.vectorstore.milvus_gateway import milvus_gateway


def count_distinct_books() -> int:
    """按书名去重统计 chunks collection 中收录的书籍数。"""
    iterator = milvus_gateway.client.query_iterator(
        collection_name=milvus_gateway.chunks_collection,
        batch_size=1000,
        output_fields=["book_name", "item_name"],
    )
    book_names: set[str] = set()
    try:
        while batch := iterator.next():
            for row in batch:
                book_name = (row.get("book_name") or row.get("item_name") or "").strip()
                if book_name:
                    book_names.add(book_name)
    finally:
        iterator.close()
    return len(book_names)
