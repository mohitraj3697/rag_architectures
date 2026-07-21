from __future__ import annotations


from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from common import build_vectorstore, format_docs, get_llm

# Long-term memory store, keyed by user_id. In production swap this dict for
# a real database (Postgres, Redis, etc.) -- the graph logic stays the same.
LONG_TERM_MEMORY: dict[str, dict] = {}


class MemoryState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    user_id: str
    question: str
    documents: list
    answer: str


def make_graph(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()

    def load_memory(state: MemoryState) -> MemoryState:
        profile = LONG_TERM_MEMORY.get(state["user_id"], {})
        note = f" (known user preferences: {profile})" if profile else ""
        return {"question": state["question"] + note}

    def retrieve(state: MemoryState) -> MemoryState:
        docs = retriever.invoke(state["question"])
        return {"documents": docs}

    def generate(state: MemoryState) -> MemoryState:
        context = format_docs(state["documents"])
        prompt = (
            "Answer using the context and any stated user preferences. "
            f"Context:\n{context}\n\nRequest: {state['question']}\nAnswer:"
        )
        response = llm.invoke(prompt)
        return {"answer": response.content, "messages": [response]}

    def update_memory(state: MemoryState) -> MemoryState:
        """Cheap heuristic extraction -- ask the LLM if a durable preference
        was stated, and if so store it in long-term memory."""
        extraction_prompt = (
            "Does this message state a durable personal preference or fact "
            "(e.g. dietary restriction, role, recurring need)? If yes, reply "
            "with a short 'key: value' pair. If no, reply with 'none'.\n\n"
            f"Message: {state['question']}"
        )
        result = llm.invoke(extraction_prompt).content.strip()
        if result.lower() != "none" and ":" in result:
            key, _, value = result.partition(":")
            profile = LONG_TERM_MEMORY.setdefault(state["user_id"], {})
            profile[key.strip()] = value.strip()
        return {}

    graph = StateGraph(MemoryState)
    graph.add_node("load_memory", load_memory)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("update_memory", update_memory)
    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "update_memory")
    graph.add_edge("update_memory", END)

    # MemorySaver gives short-term (per-thread) memory of the raw message
    # history across .invoke() calls that share the same thread_id.
    return graph.compile(checkpointer=MemorySaver())


def ask(app, user_id: str, question: str, thread_id: str = "demo-thread"):
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {"messages": [HumanMessage(content=question)], "user_id": user_id, "question": question},
        config=config,
    )
    return result["answer"]


def main():
    vectordb = build_vectorstore()
    app = make_graph(vectordb)
    user_id = "user-42"

    turn1 = ask(app, user_id, "I'm vegetarian, please remember that.")
    print("Turn 1 answer:", turn1)

    turn2 = ask(app, user_id, "Suggest something from the onboarding catering options for me.")
    print("\nTurn 2 answer:", turn2)
    print("\nLong-term memory now stored for this user:", LONG_TERM_MEMORY.get(user_id))


if __name__ == "__main__":
    main()
