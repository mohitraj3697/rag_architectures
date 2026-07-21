from __future__ import annotations


import sys
from typing import List, TypedDict

from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from common import build_vectorstore, format_docs, get_llm

TOP_K = 3


class RAGState(TypedDict):
    question: str
    documents: List[Document]
    answer: str


def make_graph(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
    llm = get_llm()

    def retrieve(state: RAGState) -> RAGState:
        docs = retriever.invoke(state["question"])
        return {"documents": docs}

    def generate(state: RAGState) -> RAGState:
        context = format_docs(state["documents"])
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the context doesn't contain the answer, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {state['question']}\nAnswer:"
        )
        response = llm.invoke(prompt)
        return {"answer": response.content}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "What are the pricing tiers?"
    vectordb = build_vectorstore()
    app = make_graph(vectordb)
    result = app.invoke({"question": question})

    print(f"\nQ: {question}\n")
    print(f"A: {result['answer']}\n")
    print("Sources:")
    for d in result["documents"]:
        print(f"  - {d.metadata.get('source')}")


if __name__ == "__main__":
    main()
