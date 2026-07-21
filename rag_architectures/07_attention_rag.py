from __future__ import annotations


import sys
from typing import List, Tuple, TypedDict

import numpy as np
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from common import build_vectorstore, get_embeddings, get_llm

FETCH_K = 6  # cast a wider net than usual, then let weighting do the filtering


def cosine_sim(a: List[float], b: List[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class AttentionState(TypedDict):
    question: str
    documents: List[Document]
    weighted: List[Tuple[Document, float]]
    answer: str


def make_graph(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": FETCH_K})
    embeddings = get_embeddings()
    llm = get_llm()

    def retrieve_multiple(state: AttentionState) -> AttentionState:
        docs = retriever.invoke(state["question"])
        return {"documents": docs}

    def attention_weighting(state: AttentionState) -> AttentionState:
        query_vec = embeddings.embed_query(state["question"])
        doc_vecs = embeddings.embed_documents([d.page_content for d in state["documents"]])
        weighted = sorted(
            zip(state["documents"], (cosine_sim(query_vec, v) for v in doc_vecs)),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return {"weighted": weighted}

    def generate(state: AttentionState) -> AttentionState:
        # Surface the weights explicitly so the LLM's own "attention" mirrors ours.
        annotated = "\n\n".join(
            f"[weight={w:.2f}] (source: {d.metadata.get('source')}) {d.page_content}"
            for d, w in state["weighted"]
        )
        prompt = (
            "Each context passage below has a relevance weight from 0 to 1. "
            "Give more weight to higher-scored passages when forming your answer, "
            "and largely ignore passages below 0.3 unless nothing else is relevant.\n\n"
            f"{annotated}\n\nQuestion: {state['question']}\nAnswer:"
        )
        return {"answer": llm.invoke(prompt).content}

    graph = StateGraph(AttentionState)
    graph.add_node("retrieve_multiple", retrieve_multiple)
    graph.add_node("attention_weighting", attention_weighting)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve_multiple")
    graph.add_edge("retrieve_multiple", "attention_weighting")
    graph.add_edge("attention_weighting", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "What causes upload errors and how are they fixed?"
    vectordb = build_vectorstore()
    app = make_graph(vectordb)
    result = app.invoke({"question": question})

    print(f"\nQ: {question}\n")
    print("Attention weights:")
    for doc, w in result["weighted"]:
        print(f"  {w:.2f}  {doc.metadata.get('source')}")
    print(f"\nA: {result['answer']}")


if __name__ == "__main__":
    main()
