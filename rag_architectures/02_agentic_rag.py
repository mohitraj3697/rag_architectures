from __future__ import annotations


import sys
from typing import Annotated, List, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from common import build_vectorstore, format_docs, get_llm

SYSTEM_PROMPT = (
    "You are a research agent. You can call the `search_knowledge_base` tool "
    "as many times as needed (with different, more specific queries) to gather "
    "facts before answering. Only answer once you have enough information. "
    "Cite which facts came from the tool in your final answer."
)


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


def make_graph(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the Nimbus Cloud knowledge base for facts relevant to `query`."""
        docs = retriever.invoke(query)
        return format_docs(docs)

    tools = [search_knowledge_base]
    llm = get_llm().bind_tools(tools)

    def agent_planner(state: AgentState) -> AgentState:
        """Decide whether to call a tool or answer, given everything so far."""
        return {"messages": [llm.invoke(state["messages"])]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_planner)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    # tools_condition routes to "tools" if the last AI message requested a
    # tool call, otherwise routes to END -- this is the "reasoning loop".
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def main():
    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Compare Nimbus Cloud's revenue growth to its churn change, and say what's driving it."
    )
    vectordb = build_vectorstore()
    app = make_graph(vectordb)

    result = app.invoke(
        {"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]}
    )

    print(f"\nQ: {question}\n")
    for m in result["messages"]:
        if m.type == "tool":
            print(f"[tool result]\n{m.content}\n")
        elif m.type == "ai" and m.content:
            print(f"[agent]\n{m.content}\n")

    print("Final answer:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
