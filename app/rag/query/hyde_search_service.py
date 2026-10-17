from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import AnnSearchRequest
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger
from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.shared.runtime.load_prompt import load_prompt


def validate_and_get_data(state:QueryGraphState) -> tuple[str,list[str]]:
    item_names:list[str] = state.get("item_names")
    rewritten_query:str  = state.get("rewritten_query")
    if (not item_names) and (not rewritten_query):
        logger.error(f"item_names或者rewritten_query为空,业务终止")
        raise ValueError(f"item_names或者rewritten_query为空,业务终止")
    return rewritten_query,item_names


def call_llm_generate_hyde_answer(rewritten_query:str)->str:
    llm_client = llm_provider.chat()
    hyde_prompt_str:str = load_prompt("hyde_prompt",rewritten_query=rewritten_query)
    messages = [
        HumanMessage(content=hyde_prompt_str)
    ]
    chains = llm_client | StrOutputParser()
    hyde_answer:str =  chains.invoke(messages)
    return hyde_answer


def rewritten_query_embedding(rewritten_query:str,hyde_answer:str) -> tuple[list,dict]:
    result = llm_provider.generate_infra_embeddings([rewritten_query + ":" + hyde_answer])
    dense_vector  = result['dense'][0]
    sparse_vector = result['sparse'][0]
    return dense_vector,sparse_vector


def milvus_hybrid_search(
    item_names: list[str] | None,
    dense_vector,
    sparse_vector,
    filter_expr: str = "",
    fallback_filter_expr: str | None = None,
) -> list[dict]:
    from app.rag.query.embedding_search_service import milvus_hybrid_search as search
    return search(
        item_names,
        dense_vector,
        sparse_vector,
        filter_expr=filter_expr,
        fallback_filter_expr=fallback_filter_expr,
    )


def search_by_hyde(state: QueryGraphState) -> QueryGraphState:
    rewritten_query,item_names = validate_and_get_data(state)
    hyde_answer:str = call_llm_generate_hyde_answer(rewritten_query)
    dense_vector,sparse_vector = rewritten_query_embedding(rewritten_query,hyde_answer)
    from app.rag.query.embedding_search_service import _content_type_relaxed_filter
    milvus_result:list[dict] = milvus_hybrid_search(
        item_names,
        dense_vector,
        sparse_vector,
        state.get("filter_expr", ""),
        _content_type_relaxed_filter(state),
    )
    if not milvus_result:
        logger.warning(f"问题:{rewritten_query} , item_names:{item_names} 没有匹配到任何数据!")
        return state
    from app.rag.query.embedding_search_service import normalize_retrieved_chunk
    embedding_chunks:list[dict] = normalize_retrieved_chunk(milvus_result)

    state['hyde_embedding_chunks'] = embedding_chunks
    return state