from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from app.infra.llm.providers import llm_provider
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import RERANK_MAX_INPUT_TOKENS, RERANK_MIN_SUMMARY_CHARS, RERANK_SUMMARY_CHAR_RATIO, \
    RERANK_MAX_TOPK, RERANK_MIN_TOPK, RERANK_GAP_RATIO, RERANK_GAP_ABS, AUDIOBOOK_RERANK_THRESHOLD
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger


def validate_and_get_data(state:QueryGraphState) -> tuple[str,list[dict],list[dict]]:
    """校验获取参数"""
    rewritten_query:str = state.get('rerank_query') or state.get('rewritten_query')
    rrf_chunks:list[dict] = state.get('rrf_chunks')
    web_search_docs:list[dict] = state.get('web_search_docs')

    if not rewritten_query:
        raise ValueError("rewritten_query cannot be empty")
    return rewritten_query, rrf_chunks or [], web_search_docs or []


def merge_rrf_web_list(rrf_chunks:list[str], web_search_docs:list[str]) -> list[dict]:
    """数据合并"""
    merge_list:list[dict] = rrf_chunks + web_search_docs
    logger.info(f"数据融合后长度:{len(merge_list)}")
    return merge_list


def create_question_answer_pair_list(rewritten_query:str, merge_list:list[dict]) -> list[list[str]]:
    """创建问题答案对儿"""
    question_answer_pair:list[list[str]] = []
    question_token_number:int = llm_provider.reranker_str_to_tokens_number(rewritten_query)

    for chunk in merge_list:
        content = chunk.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"重排文档必须是非空字串内容: {chunk.get('url') or chunk.get('chunk_id')}"
            )
        content = content.strip()
        answer_token_number:int = llm_provider.reranker_str_to_tokens_number(content)
        if (answer_token_number + question_token_number) > RERANK_MAX_INPUT_TOKENS:
            compass_char_len:int = max(
                RERANK_MIN_SUMMARY_CHARS,
                int((RERANK_MAX_INPUT_TOKENS - question_token_number) / RERANK_SUMMARY_CHAR_RATIO),
            )
            reranker_prompt:str = load_prompt(
                "rerank_text_refine",
                question=rewritten_query,
                answer=content,
                limit=compass_char_len,
            )
            messages = [
                HumanMessage(content=reranker_prompt)
            ]
            llm_client = llm_provider.chat()
            chains = llm_client | StrOutputParser()
            content = chains.invoke(messages)
        # 将问题 + content 列表添加到容器
        question_answer_pair.append([rewritten_query, content])
    return question_answer_pair


def reranker_compute_score(question_answer_pair:list[list[str]]) -> list[float]:
    score_list:list[float] = llm_provider.reranker_computer_scores(question_answer_pair)
    logger.info(f"计算得分结果:{score_list}")
    return score_list


def merge_list_add_score(merge_list:list[dict], score_list:list[float]):
    logger.info(f"赋分前数据示例:{merge_list[0]}")
    for score, chunk in zip(score_list, merge_list):
        chunk['score'] = score
    logger.info(f"赋分后数据示例:{merge_list[0]}")


def sort_merge_list(merge_list:list[dict]):
    logger.info(f"排序前数据:{merge_list}")
    merge_list.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    logger.info(f"排序后数据:{merge_list}")


def dynamic_top_k(merge_list:list[dict]) -> list[dict]:
    # 定义常量
    max_top_k = RERANK_MAX_TOPK
    min_top_k = RERANK_MIN_TOPK
    gap_ratio = RERANK_GAP_RATIO
    gap_abs = RERANK_GAP_ABS
    # 解决 max_top_k > len 导致 range 循环,index 越界问题
    max_top_k = min(len(merge_list),max_top_k)
    # 定义截取数量
    top_k = max_top_k
    if max_top_k > min_top_k:
        for index in range(min_top_k-1, max_top_k-1):
            pre_score:float = merge_list[index].get('score',0.0)
            next_score:float = merge_list[index+1].get('score',0.0)
            abs = pre_score - next_score
            ratio = abs / (pre_score + 1e-8)

            if abs > gap_abs or ratio > gap_ratio:
                top_k = index + 1
                logger.info(f"找到断崖位置,截取数量:{top_k}")
                break
    reranker_docs:list[dict] = merge_list[:top_k]
    return reranker_docs


def rerank_documents(state: QueryGraphState) -> QueryGraphState:
    # 获取并校验参数
    rewritten_query, rrf_chunks, web_search_docs = validate_and_get_data(state)
    # 合并数据
    merge_list:list[dict] = merge_rrf_web_list(rrf_chunks, web_search_docs)
    if not merge_list:
        state["reranked_docs"] = []
        state["has_item_names"] = False
        return state
    # 拼接问题\答案 对
    question_answer_pair_list:list[list[str]] = create_question_answer_pair_list(rewritten_query, merge_list)
    # 模型打分
    score_list:list[float] = reranker_compute_score(question_answer_pair_list)
    #  给 merge_list 赋值,确保原有 content
    merge_list_add_score(merge_list, score_list)
    # merge_list 排序
    sort_merge_list(merge_list)
    merge_list = [
        item for item in merge_list
        if item.get("score", 0.0) >= AUDIOBOOK_RERANK_THRESHOLD
    ]
    # 动态 top_k 截取
    reranker_docs:list[dict] = dynamic_top_k(merge_list)
    # 更新 state
    state['reranked_docs'] = reranker_docs
    return state
