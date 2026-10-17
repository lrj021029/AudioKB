"""web 搜索服务"""
import asyncio
import json
from typing import Any
import httpx
from agents.mcp import MCPServerStreamableHttp
from mcp.shared.exceptions import McpError
from app.infra.config.providers import infra_config
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger


def validate_and_get_data(state:QueryGraphState)->str:
    rewritten_query:str | None = state.get("rewritten_query")
    if not(rewritten_query):
        logger.error(f"rewritten_query is None")
        raise ValueError("rewritten_query is None")
    return rewritten_query


async def call_bailian_mcp_server(rewritten_query:str) -> Any:
    async with MCPServerStreamableHttp(
        name="Bailian MCP Server",
        params={
            "url": infra_config.mcp_config.mcp_base_url,
            "headers": {"Authorization": f"Bearer {infra_config.mcp_config.api_key}"},
            "timeout": 100,
        },
        cache_tools_list=True,
        max_retry_attempts=3
    ) as server:
        await server.connect()
        result = await server.call_tool(
            tool_name="bailian_web_search",
            arguments={
                "query": rewritten_query,
                "count": 10
            }
        )
        logger.info(f"Bailian MCP Server 搜索完成: {rewritten_query}")
        return result


def normalize_web_search_result(bailian_result:Any)->list[dict]:
    # web 搜索结果归一化
    result_str = bailian_result.content[0].text
    result_dict:dict = json.loads(result_str)
    result_list:list[dict] = result_dict.get("pages",[])
    web_search_docs: list[dict] = []
    for item in result_list:
        content = item.get("content") or item.get("snippet") or item.get("title")
        if not isinstance(content, str) or not content.strip():
            logger.warning(f"跳过缺少正文的联网检索结果: {item.get('url', '')}")
            continue
        web_search_docs.append(
            {
                "content": content.strip(),
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "type": "web",
            }
        )
    logger.info(f"搜索到{len(web_search_docs)}")
    return web_search_docs


def search_by_web(state: QueryGraphState) -> QueryGraphState:
    if not state.get("web_search_enabled", False):
        logger.info("联网检索已关闭，跳过 DashScope MCP 调用")
        state["web_search_docs"] = []
        state["web_search_error"] = ""
        return state

    rewritten_query:str = validate_and_get_data(state)
    try:
        bailian_result = asyncio.run(call_bailian_mcp_server(rewritten_query))
    except (McpError, httpx.HTTPError, TimeoutError) as exc:
        logger.opt(exception=exc).warning(
            "联网检索暂不可用，本次查询继续使用本地召回结果"
        )
        state["web_search_error"] = "联网检索暂不可用"
        state["web_search_docs"] = []
        return state
    web_search_docs:list[dict] = normalize_web_search_result(bailian_result)
    state["web_search_docs"] = web_search_docs
    state["web_search_error"] = ""
    return state