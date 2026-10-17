from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from app.infra.llm.providers import llm_provider
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.utils.task_utils import add_done_task,add_running_task,push_to_session
from app.shared.utils.sse_utils import SSEEvent
from app.infra.persistence.history_repository import history_repository
from app.rag.query.library_stats_service import count_distinct_books
import re


def state_has_item_names(state: QueryGraphState) -> tuple[bool]:
    """判断是否识别到主体"""
    item_names:list[str] = state.get("item_names",[])
    has_item_names:bool = True if item_names else False
    return has_item_names


def _load_answer_ouput_prompt(state: QueryGraphState) -> str:
    """加载提示词文本"""
    question:str = state.get("rewritten_query")
    item_names:list[str] = state.get("item_names")
    session_id:str = state.get("session_id")
    recent_message_list:list[dict] = history_repository.list_recent(session_id=session_id,limit=20)
    # 过滤有效聊天记录
    valid_message_list:list[dict] = [
        item for item in recent_message_list if item.get('item_names')
    ]
    history:str = ""
    if valid_message_list:
        ordered_messages = list(reversed(valid_message_list))
        for index in range(1, len(ordered_messages), 2):
            message_user:dict = ordered_messages[index-1]
            message_assistant:dict = ordered_messages[index]
            history += (
                f"第{(index+1)/2}轮对话:"
                f"[本轮提问] 原始内容:{message_user.get('text')},重写后的问题:{message_user.get('rewritten_query')},本次提问关联的主体:{message_user.get('item_names')}\n"
                f"[本轮回答] 原始内容:{ message_assistant.get('text')},重写后的问题:{ message_assistant.get('rewritten_query')},本次提问关联的主体:{ message_assistant.get('item_names')}\n\n"
            )
    else:
        history = "当前会话无历史聊天记录"
    context:str = ""
    reranked_docs:list[dict] = state.get("reranked_docs",[])
    intent = state.get("intent")

    for chunk in reranked_docs:
        context += (
            f"来源: {'联网检索' if chunk.get('type') == 'web' else '本地知识库'}; "
            f"书名: {chunk.get('book_name') or chunk.get('item_name', '')}; "
            f"作者: {chunk.get('author_name', '')}; "
            f"内容类型: {chunk.get('content_type', '')}; "
            f"来源文件: {chunk.get('source_file') or chunk.get('file_title', '')}; "
            f"标题: {chunk.get('title')}; 置信度: {chunk.get('score')}; "
            f"内容: {chunk.get('content')}\n\n"
        )
    state["citations"] = _build_citations(reranked_docs)
    state["sources"] = state["citations"]
    if intent == "book_recommend":
        state["recommendations"] = _build_recommendations(reranked_docs)

    prompt_template = state.get("prompt_template")
    if prompt_template == "book_recommend" or intent == "book_recommend":
        answer_output_prompt = load_prompt(
            "book_recommend",
            context=context,
            filters=state.get("filters", {}),
            question=question,
        )
    elif prompt_template == "book_detail" or intent == "book_detail":
        answer_output_prompt = load_prompt(
            "book_detail",
            context=context,
            book_name=state.get("filters", {}).get("book_name") or question,
            question=question,
        )
    else:
        answer_output_prompt = load_prompt(
            "answer_out",
            context=context,
            history=history,
            item_names=item_names,
            question=question,
        )
    return answer_output_prompt


def _build_citations(reranked_docs: list[dict]) -> list[dict]:
    citations: list[dict] = []
    seen_sources: set[tuple[str, str, str, str]] = set()
    for chunk in reranked_docs:
        citation = {
            "book_name": chunk.get("book_name") or chunk.get("item_name", ""),
            "author_name": chunk.get("author_name", ""),
            "content_type": chunk.get("content_type", ""),
            "source_file": chunk.get("source_file") or chunk.get("file_title", ""),
            "source_path": chunk.get("source_path") or (
                chunk.get("url", "") if chunk.get("type") == "web" else ""
            ),
            "title": chunk.get("title", ""),
            "url": chunk.get("url", ""),
            "type": chunk.get("type", "milvus"),
        }
        source_key = (
            str(citation["book_name"]),
            str(citation["author_name"]),
            str(citation["content_type"]),
            str(citation["source_path"] or citation["source_file"] or citation["url"]),
        )
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        citations.append(citation)
    return citations


def _build_recommendations(reranked_docs: list[dict]) -> list[dict]:
    recommendations: dict[str, dict] = {}
    for chunk in reranked_docs:
        book_name = chunk.get("book_name") or chunk.get("item_name") or ""
        if not book_name:
            continue
        recommendation = recommendations.setdefault(
            book_name,
            {
                "book_name": book_name,
                "author_name": chunk.get("author_name", ""),
                "category": chunk.get("category", ""),
                "scene_tags": chunk.get("scene_tags", ""),
                "features": [],
                "sources": [],
                "_source_keys": set(),
            },
        )
        content = chunk.get("content")
        if content and content not in recommendation["features"]:
            recommendation["features"].append(content)
        for citation in _build_citations([chunk]):
            source_key = (
                citation["book_name"],
                citation["author_name"],
                citation["content_type"],
                citation["source_path"] or citation["source_file"] or citation["url"],
            )
            if source_key not in recommendation["_source_keys"]:
                recommendation["_source_keys"].add(source_key)
                recommendation["sources"].append(citation)
    results = []
    for recommendation in recommendations.values():
        recommendation.pop("_source_keys")
        results.append(recommendation)
    return results


def call_model_answer_out(state: QueryGraphState):
    llm_client = llm_provider.chat()
    answer_out_prompt:str = _load_answer_ouput_prompt(state)
    messages = [HumanMessage(content=answer_out_prompt)]
    # 判断流式
    is_stream:bool = state.get("is_stream")
    chains = llm_client | StrOutputParser()
    answer:str = ""

    if is_stream:
        stream = chains.stream(messages)
        for content in stream:
            push_to_session(session_id=state.get("progress_id") or state.get("session_id"),event=SSEEvent.DELTA,data={"delta":content})
            answer += content
    else:
        answer:str = chains.invoke(messages)
    state["answer"] = answer


def extract_img_urls_by_reranked_docs(state: QueryGraphState) -> list[str]:
    img_urls:list[str] = []
    img_rep = re.compile(r"\!\[.*?\]\((.*?)\)")
    reranked_docs:list[dict] = state.get("reranked_docs",[])
    # 循环获取每个 chunk
    for chunk in reranked_docs:
        type_str:str = chunk.get("type")
        if type_str == 'web':
            url:str = chunk.get("url")
            if url.endswith((".png", ".jpg", ".gif", ".jpeg", ".svg")):
                img_urls.append(url)
        else:
            content:str = chunk.get("content")
            content_img_urls:list[str] = img_rep.findall(content)
            if content_img_urls:
                img_urls.extend(content_img_urls)
    state["img_urls"] = img_urls
    state["image_urls"] = img_urls


def save_assistant_chat_message(state: QueryGraphState):
    history_repository.save_message(
        session_id=state.get("session_id"),
        role='assistant',
        text=state.get("answer"),
        rewritten_query=state.get("rewritten_query"),
        item_names=state.get("item_names",[]),
        image_urls=state.get("image_urls",[]),
        citations=state.get("citations", []),
        intent=state.get("intent", ""),
    )


def generate_answer(state: QueryGraphState) -> QueryGraphState:
    reranked_docs: list[dict] = state.get("reranked_docs", [])
    if state.get("intent") == "book_count":
        book_count = count_distinct_books()
        state["answer"] = f"本地知识库当前收录 {book_count} 本书（按书名去重统计）。"
        state["citations"] = []
    elif reranked_docs:
        call_model_answer_out(state)
        extract_img_urls_by_reranked_docs(state)
    else:
        state["answer"] = "知识库中暂未找到足够的相关资料，请尝试补充书名、作者或更具体的需求。"
        state["citations"] = []
    save_assistant_chat_message(state)
    return {
        "session_id": state.get("session_id"),
        "answer": state.get("answer"),
        "image_urls": state.get("image_urls"),
        "is_stream": state.get("is_stream"),
        "citations": state.get("citations", []),
        "sources": state.get("sources", []),
        "warnings": (
            ["联网检索暂不可用，回答基于本地知识库。"]
            if state.get("web_search_error")
            else []
        ),
        "intent": state.get("intent", ""),
        "recommendations": state.get("recommendations", []),
        "book_detail": (
            state.get("reranked_docs", [])
            if state.get("intent") == "book_detail"
            else state.get("book_detail", [])
        ),
    }