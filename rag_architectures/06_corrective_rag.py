from __future__ import annotations


import sys
from typing import List, Literal, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

from common import build_vectorstore, format_docs, get_llm

RELEVANCE_THRESHOLD = 0.5  # fraction of docs that must be graded relevant


class DocGrade(BaseModel):
    relevant: bool = Field(description="Whether this single document is relevant to the question")


class CorrectiveState(TypedDict):
    question: str
    original_question: str
    documents: List[Document]
    filtered_documents: List[Document]
    relevant_ratio: float
    retries: int
    answer: str


MAX_REQUERIES = 1  # bound the re-query loop so it can't spin forever


def make_graph(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    llm = get_llm()
    grader = llm.with_structured_output(DocGrade)

    def retrieve(state: CorrectiveState) -> CorrectiveState:
        docs = retriever.invoke(state["question"])
        return {
            "documents": docs,
            "original_question": state.get("original_question") or state["question"],
            "retries": state.get("retries", 0),
        }

    def grade_and_filter(state: CorrectiveState) -> CorrectiveState:
        kept = []
        for doc in state["documents"]:
            grade = grader.invoke(
                f"Question: {state['original_question']}\nDocument: {doc.page_content}\n"
                "Is this document relevant to answering the question?"
            )
            if grade.relevant:
                kept.append(doc)
        ratio = len(kept) / max(len(state["documents"]), 1)
        return {"filtered_documents": kept, "relevant_ratio": ratio}

    def route(state: CorrectiveState) -> Literal["generate", "transform_query"]:
        if state["relevant_ratio"] >= RELEVANCE_THRESHOLD or state["retries"] >= MAX_REQUERIES:
            return "generate"
        return "transform_query"

    def transform_query(state: CorrectiveState) -> CorrectiveState:
        """Rewrite the query for a better second retrieval pass (stand-in for
        the 'web search fallback' step some CRAG implementations use)."""
        rewritten = llm.invoke(
            "The retrieved documents were mostly irrelevant. Rewrite this query "
            f"with different phrasing/keywords to retrieve better results: {state['original_question']}"
        ).content
        return {"question": rewritten, "retries": state["retries"] + 1}

    def generate(state: CorrectiveState) -> CorrectiveState:
        docs = state["filtered_documents"] or state["documents"]
        context = format_docs(docs)
        prompt = f"Context:\n{context}\n\nQuestion: {state['original_question']}\nAnswer:"
        return {"answer": llm.invoke(prompt).content}

    graph = StateGraph(CorrectiveState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_and_filter", grade_and_filter)
    graph.add_node("transform_query", transform_query)
    graph.add_node("generate", generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_and_filter")
    graph.add_conditional_edges(
        "grade_and_filter", route, {"generate": "generate", "transform_query": "transform_query"}
    )
    # After one re-query we go straight to generate (bounded retry, avoids infinite loop)
    graph.add_edge("transform_query", "retrieve")
    graph.add_edge("generate", END)
    return graph.compile()


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "Explain CRISPR gene editing."
    vectordb = build_vectorstore()
    app = make_graph(vectordb)
    # NOTE: for a demo query far outside the knowledge base (like CRISPR), this
    # will correctly show low relevance and trigger a query rewrite + retry.
    result = app.invoke({"question": question, "original_question": question, "retries": 0})

    print(f"\nQ: {question}\n")
    print(f"A: {result['answer']}\n")
    print(f"Relevant-doc ratio on final pass: {result['relevant_ratio']:.2f}")


if __name__ == "__main__":
    main()
