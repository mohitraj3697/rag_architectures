from __future__ import annotations


import sys
from typing import List, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

from common import build_vectorstore, format_docs, get_llm


class CitedAnswer(BaseModel):
    answer: str = Field(description="The final answer to the user's question")
    cited_source_ids: List[str] = Field(
        description="IDs (from the numbered context, e.g. '1', '2') of passages actually used"
    )
    reasoning_trace: str = Field(
        description="Step-by-step explanation of how the cited passages lead to the answer"
    )


class XAIState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    cited_source_ids: List[str]
    reasoning_trace: str


def make_graph(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = get_llm()
    structured_llm = llm.with_structured_output(CitedAnswer)

    def retrieve(state: XAIState) -> XAIState:
        return {"documents": retriever.invoke(state["question"])}

    def generate_with_citations(state: XAIState) -> XAIState:
        context = format_docs(state["documents"])
        prompt = (
            "Answer the question using only the numbered context passages below. "
            "You must list which passage numbers you actually relied on, and explain "
            "your reasoning step by step, referencing those numbers explicitly.\n\n"
            f"Context:\n{context}\n\nQuestion: {state['question']}"
        )
        result = structured_llm.invoke(prompt)
        return {
            "answer": result.answer,
            "cited_source_ids": result.cited_source_ids,
            "reasoning_trace": result.reasoning_trace,
        }

    graph = StateGraph(XAIState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_with_citations", generate_with_citations)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate_with_citations")
    graph.add_edge("generate_with_citations", END)
    return graph.compile()


def main():
    question = (
        sys.argv[1] if len(sys.argv) > 1 else "Why would an enterprise application be rejected?"
    )
    vectordb = build_vectorstore()
    app = make_graph(vectordb)
    result = app.invoke({"question": question})

    print(f"\nQ: {question}\n")
    print(f"A: {result['answer']}\n")
    print(f"Reasoning trace:\n{result['reasoning_trace']}\n")

    print("Cited sources:")
    for i, doc in enumerate(result["documents"], start=1):
        marker = "✓" if str(i) in result["cited_source_ids"] else " "
        print(f"  [{marker}] [{i}] {doc.metadata.get('source')}")


if __name__ == "__main__":
    main()
