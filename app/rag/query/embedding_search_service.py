from pymilvus import AnnSearchRequest
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger
from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.rag.query.config import AUDIOBOOK_DENSE_WEIGHT, AUDIOBOOK_SEARCH_TOP_K, AUDIOBOOK_SPARSE_WEIGHT
from app.shared.utils.escape_milvus_string_utils import escape_milvus_string


def validate_and_get_data(state:QueryGraphState) -> tuple[str,list[str]]:
    item_names:list[str] = state.get("item_names")
    rewritten_query:str  = state.get("rewritten_query")
    if (not item_names) and (not rewritten_query):
        logger.error(f"item_names或者rewritten_query为空,业务终止")
        raise ValueError(f"item_names或者rewritten_query为空,业务终止")
    return rewritten_query,item_names


def rewritten_query_embedding(rewritten_query:str) -> tuple[list,dict]:
    result = llm_provider.generate_infra_embeddings([rewritten_query])

    dense_vector  = result['dense'][0]
    sparse_vector = result['sparse'][0]
    return dense_vector,sparse_vector


_OUTPUT_FIELDS = [
    "chunk_id", "title", "parent_title", "part", "file_title", "item_name",
    "content", "content_type", "book_name", "author_name", "category",
    "scene_tags", "duration", "source_file", "source_path",
]


def _combine_filter(item_names: list[str] | None, filter_expr: str = "") -> str:
    expressions: list[str] = []
    names = [name for name in (item_names or []) if name]
    if names:
        escaped_names = ", ".join(f'"{escape_milvus_string(name)}"' for name in names)
        expressions.append(f"item_name in [{escaped_names}]")
    if filter_expr.strip():
        expressions.append(f"({filter_expr})")
    return " and ".join(expressions)


def milvus_hybrid_search(
    item_names: list[str] | None,
    dense_vector,
    sparse_vector,
    filter_expr: str = "",
    limit: int = AUDIOBOOK_SEARCH_TOP_K,
    fallback_filter_expr: str | None = None,
) -> list[dict]:
    expr_str = _combine_filter(item_names, filter_expr)
    reqs:list[AnnSearchRequest]  = milvus_gateway.create_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        limit=2 * limit,
        expr=expr_str or None,
    )
    resp = milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.chunks_collection,
        reqs=reqs,
        ranker_weights=(AUDIOBOOK_DENSE_WEIGHT,AUDIOBOOK_SPARSE_WEIGHT),
        norm_score=True,
        limit=limit,
        output_fields=_OUTPUT_FIELDS,
    )

    if resp and resp[0]:
        logger.info(f"完成数据查询,共查询到数据:{len(resp[0])}")
        return resp[0]
    if fallback_filter_expr is not None and fallback_filter_expr != filter_expr:
        fallback_expr_str = _combine_filter(item_names, fallback_filter_expr)
        logger.warning(
            "带内容类型约束的检索未命中，保留书名/其他过滤条件并移除内容类型约束后重试"
        )
        fallback_reqs: list[AnnSearchRequest] = milvus_gateway.create_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=2 * limit,
            expr=fallback_expr_str or None,
        )
        fallback_resp = milvus_gateway.hybrid_search(
            collection_name=milvus_gateway.chunks_collection,
            reqs=fallback_reqs,
            ranker_weights=(AUDIOBOOK_DENSE_WEIGHT, AUDIOBOOK_SPARSE_WEIGHT),
            norm_score=True,
            limit=limit,
            output_fields=_OUTPUT_FIELDS,
        )
        if fallback_resp and fallback_resp[0]:
            logger.info(f"放宽内容类型约束后检索到数据:{len(fallback_resp[0])}")
            return fallback_resp[0]
    logger.warning(f"没有查询到任何数据")
    return []


def _content_type_relaxed_filter(state: QueryGraphState) -> str:
    filters = state.get("filters") or {}
    if "content_type" not in filters:
        return state.get("filter_expr", "")
    from app.rag.query.filter_build_service import build_filter_expression

    relaxed_filters = {
        key: value for key, value in filters.items() if key != "content_type"
    }
    return build_filter_expression(relaxed_filters)


def normalize_retrieved_chunk(chunks:list[dict]) -> list[dict]:
    embedding_chunks:list[dict] = [
        {
            "chunk_id": chunk.get("entity",{}).get("chunk_id"),
            "score": chunk.get("distance",0.0),
            "title": chunk.get("entity",{}).get("title"),
            "parent_title": chunk.get("entity",{}).get("parent_title"),
            "part": chunk.get("entity",{}).get("part"),
            "file_title": chunk.get("entity",{}).get("file_title"),
            "item_name": chunk.get("entity",{}).get("item_name"),
            "content": chunk.get("entity",{}).get("content"),
            "content_type": chunk.get("entity",{}).get("content_type", ""),
            "book_name": chunk.get("entity",{}).get("book_name", ""),
            "author_name": chunk.get("entity",{}).get("author_name", ""),
            "category": chunk.get("entity",{}).get("category", ""),
            "scene_tags": chunk.get("entity",{}).get("scene_tags", ""),
            "duration": chunk.get("entity",{}).get("duration", ""),
            "source_file": chunk.get("entity",{}).get("source_file", chunk.get("entity",{}).get("file_title", "")),
            "source_path": chunk.get("entity",{}).get("source_path", ""),
            "type": "milvus"
        }
        for chunk in chunks
    ]
    # 4.2 返回对应结果
    if embedding_chunks:
        logger.info(f"完成结果归一化，数据总量:{len(embedding_chunks)}")
    return embedding_chunks


def search_domain_content(query: str, filter_expr: str = "", top_k: int = AUDIOBOOK_SEARCH_TOP_K) -> list[dict]:
    if not query.strip():
        raise ValueError("query cannot be empty")
    vectors = llm_provider.generate_infra_embeddings([query])
    results = milvus_hybrid_search(
        None,
        vectors["dense"][0],
        vectors["sparse"][0],
        filter_expr=filter_expr,
        limit=top_k,
    )
    return normalize_retrieved_chunk(results)


def search_by_embedding(state: QueryGraphState) -> QueryGraphState:
    # 获取参数和校验
    rewritten_query,item_names = validate_and_get_data(state)
    # 重写问题进行向量化处理
    dense_vector,sparse_vector = rewritten_query_embedding(rewritten_query)
    filter_expr = state.get("filter_expr", "")
    # 进行数据混合检索
    milvus_result:list[dict] = milvus_hybrid_search(
        item_names,
        dense_vector,
        sparse_vector,
        filter_expr=filter_expr,
        fallback_filter_expr=_content_type_relaxed_filter(state),
    )
    # 进行结果归一化处理
    if not milvus_result:
        logger.warning(f"问题:{rewritten_query} , item_names:{item_names} 没有匹配到任何数据!")
        return state
    embedding_chunks:list[dict] = normalize_retrieved_chunk(milvus_result)
    # 修改state
    state['embedding_chunks'] = embedding_chunks
    # 返回结果
    return state