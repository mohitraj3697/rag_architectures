from __future__ import annotations


import sys
from typing import List, Literal, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

from common import build_vectorstore, format_docs, get_llm

MAX_RETRIES = 2


class RelevanceGrade(BaseModel):
    relevant: bool = Field(description="True if the documents are sufficient to answer the question")
    reasoning: str = Field(description="One sentence explanation")


class AnswerGrade(BaseModel):
    grounded: bool = Field(description="True if every claim in the answer is supported by the context")
    sufficient: bool = Field(description="True if the answer fully addresses the question")


class SelfRAGState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    retries: int
    verdict: str  # for logging


def make_graph(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = get_llm()
    relevance_grader = llm.with_structured_output(RelevanceGrade)
    answer_grader = llm.with_structured_output(AnswerGrade)

    def retrieve(state: SelfRAGState) -> SelfRAGState:
        docs = retriever.invoke(state["question"])
        return {"documents": docs, "retries": state.get("retries", 0)}

    def grade_documents(state: SelfRAGState) -> SelfRAGState:
        grade = relevance_grader.invoke(
            f"Question: {state['question']}\n\nDocuments:\n{format_docs(state['documents'])}\n\n"
            "Are these documents relevant and sufficient to answer the question?"
        )
        return {"verdict": "relevant" if grade.relevant else f"irrelevant: {grade.reasoning}"}

    def route_after_grading(state: SelfRAGState) -> Literal["generate", "rewrite_and_retry"]:
        if state["verdict"] == "relevant" or state["retries"] >= MAX_RETRIES:
            return "generate"
        return "rewrite_and_retry"

    def rewrite_and_retry(state: SelfRAGState) -> SelfRAGState:
        rewritten = llm.invoke(
            f"Rewrite this question to be more specific and retrieval-friendly: {state['question']}"
        ).content
        return {"question": rewritten, "retries": state["retries"] + 1}

    def generate(state: SelfRAGState) -> SelfRAGState:
        context = format_docs(state["documents"])
        prompt = (
            f"Context:\n{context}\n\nQuestion: {state['question']}\n"
            "Answer using only the context, and only state things the context supports."
        )
        return {"answer": llm.invoke(prompt).content}

    def critique_answer(state: SelfRAGState) -> SelfRAGState:
        grade = answer_grader.invoke(
            f"Context:\n{format_docs(state['documents'])}\n\nAnswer:\n{state['answer']}\n\n"
            "Is this answer grounded in the context and sufficient?"
        )
        ok = grade.grounded and grade.sufficient
        return {"verdict": "final" if (ok or state["retries"] >= MAX_RETRIES) else "retry_generation"}

    def route_after_critique(state: SelfRAGState) -> Literal["retrieve", "__end__"]:
        if state["verdict"] == "retry_generation":
            return "retrieve"
        return END

    graph = StateGraph(SelfRAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_and_retry", rewrite_and_retry)
    graph.add_node("generate", generate)
    graph.add_node("critique_answer", critique_answer)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents", route_after_grading, {"generate": "generate", "rewrite_and_retry": "rewrite_and_retry"}
    )
    graph.add_edge("rewrite_and_retry", "retrieve")
    graph.add_edge("generate", "critique_answer")
    graph.add_conditional_edges("critique_answer", route_after_critique, {"retrieve": "retrieve", END: END})

    return graph.compile()


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "Explain how multipart upload works."
    vectordb = build_vectorstore()
    app = make_graph(vectordb)
    result = app.invoke({"question": question, "retries": 0})

    print(f"\nQ: {question}\n")
    print(f"A: {result['answer']}\n")
    print(f"Retries used: {result['retries']}")


if __name__ == "__main__":
    main()
