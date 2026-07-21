from __future__ import annotations


import sys
from typing import List, Literal, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

from common import build_vectorstore, format_docs, get_llm


class ComplexityDecision(BaseModel):
    complexity: Literal["simple", "complex"] = Field(
        description="'simple' if answerable from general knowledge with no lookup, "
        "'complex' if it needs retrieval and multiple reasoning steps"
    )


class AdaptiveState(TypedDict):
    question: str
    complexity: str
    sub_queries: List[str]
    documents: List[Document]
    answer: str


def make_graph(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()
    classifier = llm.with_structured_output(ComplexityDecision)

    def check_complexity(state: AdaptiveState) -> AdaptiveState:
        decision = classifier.invoke(
            "Classify this question as 'simple' (general knowledge, no lookup needed) "
            f"or 'complex' (needs specific facts and multi-step reasoning): {state['question']}"
        )
        return {"complexity": decision.complexity}

    def route(state: AdaptiveState) -> Literal["direct_answer", "decompose"]:
        return "direct_answer" if state["complexity"] == "simple" else "decompose"

    def direct_answer(state: AdaptiveState) -> AdaptiveState:
        return {"answer": llm.invoke(state["question"]).content, "documents": []}

    def decompose(state: AdaptiveState) -> AdaptiveState:
        """Break a complex question into retrieval-friendly sub-queries."""
        raw = llm.invoke(
            "Break this question into 2-3 focused sub-questions, one per line, no numbering:\n"
            f"{state['question']}"
        ).content
        sub_queries = [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]
        return {"sub_queries": sub_queries[:3] or [state["question"]]}

    def multi_retrieve(state: AdaptiveState) -> AdaptiveState:
        all_docs: List[Document] = []
        seen_ids = set()
        for q in state["sub_queries"]:
            for d in retriever.invoke(q):
                if d.metadata.get("id") not in seen_ids:
                    seen_ids.add(d.metadata.get("id"))
                    all_docs.append(d)
        return {"documents": all_docs}

    def generate(state: AdaptiveState) -> AdaptiveState:
        context = format_docs(state["documents"]) if state["documents"] else "(no retrieval needed)"
        prompt = f"Context:\n{context}\n\nQuestion: {state['question']}\nAnswer thoroughly:"
        return {"answer": llm.invoke(prompt).content}

    graph = StateGraph(AdaptiveState)
    graph.add_node("check_complexity", check_complexity)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("decompose", decompose)
    graph.add_node("multi_retrieve", multi_retrieve)
    graph.add_node("generate", generate)

    graph.add_edge(START, "check_complexity")
    graph.add_conditional_edges(
        "check_complexity", route, {"direct_answer": "direct_answer", "decompose": "decompose"}
    )
    graph.add_edge("direct_answer", END)
    graph.add_edge("decompose", "multi_retrieve")
    graph.add_edge("multi_retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "What is 2+2?"
    vectordb = build_vectorstore()
    app = make_graph(vectordb)
    result = app.invoke({"question": question, "sub_queries": [], "documents": []})

    print(f"\nQ: {question}")
    print(f"Routed as: {result['complexity']}\n")
    print(f"A: {result['answer']}")


if __name__ == "__main__":
    main()
