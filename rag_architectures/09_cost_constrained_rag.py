from __future__ import annotations


import argparse
from typing import List, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from common import GROQ_MODEL, GROQ_MODEL_LARGE, build_vectorstore, format_docs, get_llm

LOW_BUDGET_THRESHOLD = 400  # below this many tokens remaining, go cheap


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) -- replace with real usage_metadata in production."""
    return max(1, len(text) // 4)


class CostState(TypedDict):
    question: str
    budget_remaining: int
    tier: str  # "full" or "economy"
    k: int
    model: str
    documents: List[Document]
    answer: str
    tokens_spent: int


def make_graph(vectordb):
    def check_budget(state: CostState) -> CostState:
        if state["budget_remaining"] >= LOW_BUDGET_THRESHOLD:
            return {"tier": "full", "k": 4, "model": GROQ_MODEL_LARGE}
        return {"tier": "economy", "k": 2, "model": GROQ_MODEL}

    def retrieve(state: CostState) -> CostState:
        retriever = vectordb.as_retriever(search_kwargs={"k": state["k"]})
        return {"documents": retriever.invoke(state["question"])}

    def generate(state: CostState) -> CostState:
        context = format_docs(state["documents"])
        if state["tier"] == "economy":
            prompt = f"Briefly, using this context, answer: {state['question']}\nContext:\n{context}"
        else:
            prompt = (
                f"Using this context, give a thorough, well-structured answer to: "
                f"{state['question']}\n\nContext:\n{context}"
            )
        llm = get_llm(model=state["model"])
        response = llm.invoke(prompt)
        spent = estimate_tokens(prompt) + estimate_tokens(response.content)
        return {
            "answer": response.content,
            "tokens_spent": spent,
            "budget_remaining": state["budget_remaining"] - spent,
        }

    graph = StateGraph(CostState)
    graph.add_node("check_budget", check_budget)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "check_budget")
    graph.add_edge("check_budget", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default="Summarize the security policy.")
    parser.add_argument("--budget", type=int, default=1000, help="starting token budget")
    args = parser.parse_args()

    vectordb = build_vectorstore()
    app = make_graph(vectordb)
    result = app.invoke({"question": args.question, "budget_remaining": args.budget})

    print(f"\nQ: {args.question}\n")
    print(f"Tier used: {result['tier']}  |  model: {result['model']}  |  k: {result['k']}")
    print(f"Tokens spent (est.): {result['tokens_spent']}  |  budget left: {result['budget_remaining']}\n")
    print(f"A: {result['answer']}")


if __name__ == "__main__":
    main()
