from langgraph.graph import END, StateGraph

from app.process.query.agent.nodes.node_answer_output import node_answer_output
from app.process.query.agent.nodes.node_book_detail import node_book_detail
from app.process.query.agent.nodes.node_book_recommend import node_book_recommend
from app.process.query.agent.nodes.node_filter_build import node_filter_build
from app.process.query.agent.nodes.node_intent_classify import node_intent_classify
from app.process.query.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.process.query.agent.nodes.node_query_entry import node_query_entry
from app.process.query.agent.nodes.node_rerank import node_rerank
from app.process.query.agent.nodes.node_rrf import node_rrf
from app.process.query.agent.nodes.node_search_embedding import node_search_embedding
from app.process.query.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.process.query.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.process.query.agent.state import QueryGraphState

query_graph_builder = StateGraph(state_schema=QueryGraphState)

query_graph_builder.add_node("entry", node_query_entry)
query_graph_builder.add_node("intent_classify", node_intent_classify)
query_graph_builder.add_node("filter_build", node_filter_build)
query_graph_builder.add_node("book_recommend", node_book_recommend)
query_graph_builder.add_node("book_detail", node_book_detail)
query_graph_builder.add_node("item_name_confirm", node_item_name_confirm)
query_graph_builder.add_node("search_embed", node_search_embedding)
query_graph_builder.add_node("search_hyde", node_search_embedding_hyde)
query_graph_builder.add_node("web_search", node_web_search_mcp)
query_graph_builder.add_node("rrf", node_rrf)
query_graph_builder.add_node("rerank", node_rerank)
query_graph_builder.add_node("answer_output", node_answer_output)


def route_by_intent(state: QueryGraphState) -> str:
    intent = state.get("intent", "general_qa")
    if intent == "book_recommend":
        return "book_recommend"
    if intent == "book_detail":
        return "book_detail"
    return "general_qa"


query_graph_builder.set_entry_point("entry")
query_graph_builder.add_edge("entry", "intent_classify")
query_graph_builder.add_edge("intent_classify", "filter_build")
query_graph_builder.add_conditional_edges(
    "filter_build",
    route_by_intent,
    {
        "book_recommend": "book_recommend",
        "book_detail": "book_detail",
        "general_qa": "item_name_confirm",
    },
)

query_graph_builder.add_edge("book_recommend", "item_name_confirm")
query_graph_builder.add_edge("book_detail", "item_name_confirm")

query_graph_builder.add_edge("item_name_confirm", "search_embed")
query_graph_builder.add_edge("item_name_confirm", "search_hyde")
query_graph_builder.add_edge("item_name_confirm", "web_search")

# RRF waits for both local retrieval branches; web results never enter RRF.
query_graph_builder.add_edge(["search_embed", "search_hyde"], "rrf")
query_graph_builder.add_edge(["rrf", "web_search"], "rerank")
query_graph_builder.add_edge("rerank", "answer_output")
query_graph_builder.add_edge("answer_output", END)

query_graph_app = query_graph_builder.compile()