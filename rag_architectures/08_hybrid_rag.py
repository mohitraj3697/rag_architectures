from __future__ import annotations


import sys
from typing import List, TypedDict

import networkx as nx
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from common import build_vectorstore, format_docs, get_llm


def build_knowledge_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edge("Nimbus Cloud", "Orbit Holdings", relation="owned_by")
    g.add_edge("Beacon Analytics", "Orbit Holdings", relation="owned_by")
    g.add_edge("Drift Messaging", "Orbit Holdings", relation="owned_by")
    g.add_edge("Orbit Holdings", "Maria Ferreira", relation="ceo")
    g.add_edge("Orbit Holdings", "2015", relation="founded_in")
    return g


def kg_facts_for_entities(g: nx.DiGraph, entities: List[str]) -> List[str]:
    facts = []
    for node in g.nodes:
        if any(e.lower() in node.lower() or node.lower() in e.lower() for e in entities):
            for _, target, data in g.out_edges(node, data=True):
                facts.append(f"{node} --[{data['relation']}]--> {target}")
            for source, _, data in g.in_edges(node, data=True):
                facts.append(f"{source} --[{data['relation']}]--> {node}")
    return sorted(set(facts))


class HybridState(TypedDict):
    question: str
    entities: List[str]
    kg_facts: List[str]
    documents: List[Document]
    answer: str


def make_graph(vectordb, kg: nx.DiGraph):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()

    def extract_entities(state: HybridState) -> HybridState:
        raw = llm.invoke(
            "List the company or person names mentioned or implied in this question, "
            f"comma-separated, nothing else: {state['question']}"
        ).content
        entities = [e.strip() for e in raw.split(",") if e.strip()]
        return {"entities": entities}

    def kg_lookup(state: HybridState) -> HybridState:
        return {"kg_facts": kg_facts_for_entities(kg, state["entities"])}

    def vector_retrieve(state: HybridState) -> HybridState:
        return {"documents": retriever.invoke(state["question"])}

    def structured_reasoning(state: HybridState) -> HybridState:
        kg_block = "\n".join(state["kg_facts"]) or "(no graph facts found)"
        doc_block = format_docs(state["documents"])
        prompt = (
            "You have two sources of truth: a knowledge graph (structured, exact) and "
            "retrieved documents (unstructured, descriptive). Prefer the knowledge graph "
            "for relationships/ownership/roles; use the documents for context and detail.\n\n"
            f"Knowledge graph facts:\n{kg_block}\n\nDocuments:\n{doc_block}\n\n"
            f"Question: {state['question']}\nAnswer, and briefly note which source each part came from:"
        )
        return {"answer": llm.invoke(prompt).content}

    graph = StateGraph(HybridState)
    graph.add_node("extract_entities", extract_entities)
    graph.add_node("kg_lookup", kg_lookup)
    graph.add_node("vector_retrieve", vector_retrieve)
    graph.add_node("structured_reasoning", structured_reasoning)

    graph.add_edge(START, "extract_entities")
    graph.add_edge("extract_entities", "kg_lookup")
    graph.add_edge("kg_lookup", "vector_retrieve")
    graph.add_edge("vector_retrieve", "structured_reasoning")
    graph.add_edge("structured_reasoning", END)
    return graph.compile()


def main():
    question = (
        sys.argv[1] if len(sys.argv) > 1 else "Who is the CEO of the company that owns Nimbus Cloud?"
    )
    vectordb = build_vectorstore()
    kg = build_knowledge_graph()
    app = make_graph(vectordb, kg)
    result = app.invoke({"question": question})

    print(f"\nQ: {question}\n")
    print("Entities detected:", result["entities"])
    print("KG facts used:", result["kg_facts"])
    print(f"\nA: {result['answer']}")


if __name__ == "__main__":
    main()
