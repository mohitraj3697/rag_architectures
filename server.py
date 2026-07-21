"""FastAPI Backend Server for 10 RAG Architectures."""

from __future__ import annotations

import importlib
import os
import sys
import time
import traceback
from functools import lru_cache
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure rag_architectures is importable
RAG_DIR = os.path.join(os.path.dirname(__file__), "rag_architectures")
sys.path.insert(0, RAG_DIR)
os.chdir(RAG_DIR)

load_dotenv(os.path.join(RAG_DIR, ".env"))
load_dotenv()

app = FastAPI(title="10 RAG Architectures API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Lazy-loaded shared resources
@lru_cache(maxsize=1)
def _get_common():
    return importlib.import_module("common")


@lru_cache(maxsize=1)
def _get_default_vectorstore():
    common = _get_common()
    return common.build_vectorstore()


_custom_vectorstore = None


def get_vector_db():
    global _custom_vectorstore
    if _custom_vectorstore is not None:
        return _custom_vectorstore
    return _get_default_vectorstore()


# Architecture metadata
ARCHITECTURES: Dict[str, Dict[str, Any]] = {

    "01_standard_rag": {
        "title": "01. Standard RAG",
        "subtitle": "The Baseline Vector Search and Context Generation Setup",
        "module": "01_standard_rag",
        "badges": [{"label": "Baseline", "color": "blue"}, {"label": "Fast", "color": "green"}, {"label": "Static KB", "color": "amber"}],
        "default_query": "What are the pricing tiers for Nimbus Cloud?",
        "slide_example": "What are the side effects of ibuprofen?",
        "how_it_works": [
            "User query is embedded into a dense vector space.",
            "Top-K nearest document chunks are retrieved from vector storage.",
            "Retrieved text context is injected into the LLM prompt.",
            "LLM generates a grounded answer strictly using the context."
        ],
        "best_when": "Knowledge is static or slow-changing, no multi-step reasoning required.",
        "usage": "FAQ bots, Knowledge base assistants, Documentation search.",
        "mermaid": """graph LR
    User([User Query]) --> Embed[Embedding Model]
    Embed --> VDB[(Chroma Vector DB)]
    VDB -->|Top-K Docs| Context[Context Injection]
    User --> Context
    Context --> LLM[Groq LLM]
    LLM --> Answer([Final Answer])
    style User fill:#1e293b,stroke:#38bdf8,color:#f8fafc
    style VDB fill:#0f172a,stroke:#818cf8,color:#f8fafc
    style LLM fill:#1e293b,stroke:#4ade80,color:#f8fafc
    style Answer fill:#0f172a,stroke:#38bdf8,color:#f8fafc
    style Embed fill:#1e293b,stroke:#38bdf8,color:#f8fafc
    style Context fill:#1e293b,stroke:#38bdf8,color:#f8fafc"""
    },
    "02_agentic_rag": {
        "title": "02. Agentic RAG",
        "subtitle": "RAG with Autonomous Tool-Calling and Iterative Reasoning Loops",
        "module": "02_agentic_rag",
        "badges": [{"label": "Autonomous", "color": "purple"}, {"label": "Tool Use", "color": "blue"}, {"label": "Multi-Step", "color": "green"}],
        "default_query": "Compare Nimbus Cloud's revenue growth to its churn change, and explain what's driving it.",
        "slide_example": "Compare Tesla and BYD revenue growth over 3 years.",
        "how_it_works": [
            "Agent evaluates user query and formulates a search plan.",
            "Agent executes tool calls (search_knowledge_base) as needed.",
            "Tool results feed back into the agent reasoning loop.",
            "Process repeats until agent has sufficient facts for the final response."
        ],
        "best_when": "Multi-step reasoning and query decomposition are needed across external tools or APIs.",
        "usage": "Research assistants, Financial analysis, Multi-source QA.",
        "mermaid": """graph TD
    User([User Query]) --> Agent[Agent / Planner]
    Agent -->|Decide Action| Decision{Need Tools?}
    Decision -->|Yes| Tools[Search Tool Call]
    Tools -->|Return Facts| Agent
    Decision -->|No / Complete| Answer([Final Answer])
    style Agent fill:#1e293b,stroke:#c084fc,color:#f8fafc
    style Tools fill:#0f172a,stroke:#38bdf8,color:#f8fafc
    style Answer fill:#1e293b,stroke:#4ade80,color:#f8fafc
    style User fill:#0f172a,stroke:#38bdf8,color:#f8fafc
    style Decision fill:#0f172a,stroke:#fbbf24,color:#f8fafc"""
    },
    "03_memory_rag": {
        "title": "03. RAG with Memory",
        "subtitle": "Short-Term Thread History + Long-Term User Profile Preference Tracking",
        "module": "03_memory_rag",
        "badges": [{"label": "Conversational", "color": "purple"}, {"label": "Personalized", "color": "green"}, {"label": "Stateful", "color": "blue"}],
        "default_query": "Suggest something from the onboarding catering options for me.",
        "slide_example": "I'm a vegetarian -> Suggest high-protein foods.",
        "how_it_works": [
            "System loads short-term thread history (via MemorySaver) and long-term user profile.",
            "User preferences are merged into the query before vector search.",
            "Retrieved documents and user history guide LLM generation.",
            "LLM automatically extracts and updates durable long-term preferences."
        ],
        "best_when": "Personalization matters and conversations span multiple turns across sessions.",
        "usage": "Personal assistants, Recommendation engines, Conversational companions.",
        "mermaid": """graph LR
    User([User Query]) --> MemRead[Load Memory Store]
    MemRead --> QueryMerge[Merge Query + Preferences]
    QueryMerge --> Retrieve[Vector DB Search]
    Retrieve --> LLM[LLM Generator]
    LLM --> MemWrite[Extract & Update Memory]
    LLM --> Answer([Personalized Answer])
    style MemRead fill:#0f172a,stroke:#c084fc,color:#f8fafc
    style MemWrite fill:#0f172a,stroke:#c084fc,color:#f8fafc
    style LLM fill:#1e293b,stroke:#4ade80,color:#f8fafc
    style User fill:#0f172a,stroke:#38bdf8,color:#f8fafc
    style QueryMerge fill:#1e293b,stroke:#38bdf8,color:#f8fafc
    style Retrieve fill:#1e293b,stroke:#38bdf8,color:#f8fafc
    style Answer fill:#0f172a,stroke:#38bdf8,color:#f8fafc"""
    },
    "04_self_rag": {
        "title": "04. Self RAG",
        "subtitle": "Self-Critiquing Retrieval Relevance, Groundedness and Query Rewriting",
        "module": "04_self_rag",
        "badges": [{"label": "Self-Critique", "color": "amber"}, {"label": "High Accuracy", "color": "green"}, {"label": "Query Rewrite", "color": "purple"}],
        "default_query": "Explain how multipart upload works step by step.",
        "slide_example": "Explain quantum entanglement simply.",
        "how_it_works": [
            "Initial document retrieval is graded for relevance by a structured grader.",
            "If documents are inadequate, the query is rewritten and re-retrieved.",
            "Generated answer is evaluated for groundedness against sources.",
            "Retries until answer quality passes critique."
        ],
        "best_when": "Answer quality matters more than speed and built-in validation is required.",
        "usage": "High-accuracy QA, Legal and Medical assistants, Scientific research.",
        "mermaid": """graph TD
    User([User Query]) --> Retrieve[Retrieve Docs]
    Retrieve --> GradeDocs{Grade Relevance}
    GradeDocs -->|Irrelevant| Rewrite[Rewrite Query]
    Rewrite --> Retrieve
    GradeDocs -->|Relevant| Generate[Generate Answer]
    Generate --> Critique{Critique Groundedness}
    Critique -->|Ungrounded| Generate
    Critique -->|Passed| Answer([Final Validated Answer])
    style GradeDocs fill:#0f172a,stroke:#fbbf24,color:#f8fafc
    style Critique fill:#0f172a,stroke:#fbbf24,color:#f8fafc
    style Answer fill:#1e293b,stroke:#4ade80,color:#f8fafc
    style User fill:#0f172a,stroke:#38bdf8,color:#f8fafc
    style Retrieve fill:#1e293b,stroke:#38bdf8,color:#f8fafc
    style Rewrite fill:#1e293b,stroke:#c084fc,color:#f8fafc
    style Generate fill:#1e293b,stroke:#38bdf8,color:#f8fafc"""
    },
    "05_adaptive_rag": {
        "title": "05. Adaptive RAG",
        "subtitle": "Dynamic Query Complexity Routing and Multi-Subquery Decomposition",
        "module": "05_adaptive_rag",
        "badges": [{"label": "Dynamic Routing", "color": "blue"}, {"label": "Query Decomposition", "color": "purple"}, {"label": "Cost Efficient", "color": "green"}],
        "default_query": "Why did Nimbus Cloud's churn drop and how does that relate to security certification?",
        "slide_example": "Simple: Capital of France? vs Complex: Impact of inflation on emerging markets?",
        "how_it_works": [
            "Classifier categorizes question as simple or complex.",
            "Simple queries bypass vector search for fast, low-cost direct LLM generation.",
            "Complex queries are decomposed into 2-3 targeted sub-queries.",
            "Sub-queries fetch parallel vector results, merged before final synthesis."
        ],
        "best_when": "Handling high query volumes with mixed simple and complex requirements.",
        "usage": "Enterprise SaaS assistants, Customer service routing, Cost-optimized QA.",
        "mermaid": """graph TD
    User([User Query]) --> Router{Complexity Check}
    Router -->|Simple| Direct[Direct LLM Answer]
    Router -->|Complex| Decompose[Sub-query Decomposition]
    Decompose --> MultiRetrieve[Parallel Multi-Retrieval]
    MultiRetrieve --> Generate[Synthesize Answer]
    Direct --> Answer([Final Answer])
    Generate --> Answer
    style Router fill:#0f172a,stroke:#38bdf8,color:#f8fafc
    style Decompose fill:#1e293b,stroke:#c084fc,color:#f8fafc
    style Answer fill:#1e293b,stroke:#4ade80,color:#f8fafc
    style User fill:#0f172a,stroke:#38bdf8,color:#f8fafc
    style Direct fill:#1e293b,stroke:#4ade80,color:#f8fafc
    style MultiRetrieve fill:#1e293b,stroke:#38bdf8,color:#f8fafc
    style Generate fill:#1e293b,stroke:#38bdf8,color:#f8fafc"""
    },
    "06_corrective_rag": {
        "title": "06. Corrective RAG (CRAG)",
        "subtitle": "Document Quality Filtering, Threshold Evaluation and Fallback Re-querying",
        "module": "06_corrective_rag",
        "badges": [{"label": "Quality Filtering", "color": "amber"}, {"label": "Noise Tolerant", "color": "green"}, {"label": "Fallback Logic", "color": "purple"}],
        "default_query": "Explain CRISPR gene editing and its applications.",
        "slide_example": "Low vector relevance triggers fallback transform and search.",
        "how_it_works": [
            "Vector search retrieves candidate documents.",
            "Document grader evaluates each document individually for relevance.",
            "If relevant ratio falls below threshold, query is transformed for fallback retrieval.",
            "Only high-quality filtered context is passed to generation."
        ],
        "best_when": "Data sources are noisy, unstructured, or retrieval quality is inconsistent.",
        "usage": "Enterprise search, Unstructured data lakes, Inconsistent document stores.",
        "mermaid": """graph TD
    User([User Query]) --> Retrieve[Retrieve Candidate Docs]
    Retrieve --> Filter[Grade & Filter Individual Docs]
    Filter --> Check{Relevant Ratio >= Threshold?}
    Check -->|No| Transform[Transform / Rewrite Query]
    Transform --> Retrieve
    Check -->|Yes| Generate[Generate with Filtered Context]
    Generate --> Answer([Final Answer])
    style Filter fill:#0f172a,stroke:#fbbf24,color:#f8fafc
    style Transform fill:#1e293b,stroke:#c084fc,color:#f8fafc
    style Answer fill:#1e293b,stroke:#4ade80,color:#f8fafc
    style User fill:#0f172a,stroke:#38bdf8,color:#f8fafc
    style Retrieve fill:#1e293b,stroke:#38bdf8,color:#f8fafc
    style Check fill:#0f172a,stroke:#fbbf24,color:#f8fafc
    style Generate fill:#1e293b,stroke:#38bdf8,color:#f8fafc"""
    },
    "07_attention_rag": {
        "title": "07. Attention-based RAG",
        "subtitle": "Cosine Embedding Vector Weighting and High-Signal Context Prioritization",
        "module": "07_attention_rag",
        "badges": [{"label": "Vector Attention", "color": "purple"}, {"label": "Relevance Weighting", "color": "blue"}, {"label": "Noise Filtering", "color": "green"}],
        "default_query": "What causes upload errors and how are they fixed?",
        "slide_example": "Causes of climate change? Emphasizes key sections like greenhouse gases.",
        "how_it_works": [
            "Casts a wider retrieval net (FETCH_K = 6).",
            "Calculates cosine similarity attention weights between query and chunk vectors.",
            "Annotates passages with numeric attention weights [weight=0.XX].",
            "LLM focuses generation heavily on high-signal content."
        ],
        "best_when": "Retrieved context is large or noisy, and fine-grained passage relevance is crucial.",
        "usage": "Long-document QA, Summarization systems, Legal and Research analysis.",
        "mermaid": """graph LR
    User([User Query]) --> Fetch[Wider Net Retrieval - Top 6]
    Fetch --> Attn[Compute Cosine Attention Weights]
    Attn --> Rank[Order & Annotate Passages]
    Rank --> LLM[LLM Generation - Weighted Prompt]
    LLM --> Answer([Weighted Answer])
    style Attn fill:#0f172a,stroke:#c084fc,color:#f8fafc
    style Rank fill:#1e293b,stroke:#38bdf8,color:#f8fafc
    style Answer fill:#1e293b,stroke:#4ade80,color:#f8fafc
    style User fill:#0f172a,stroke:#38bdf8,color:#f8fafc
    style Fetch fill:#1e293b,stroke:#38bdf8,color:#f8fafc
    style LLM fill:#1e293b,stroke:#4ade80,color:#f8fafc"""
    },
    "08_hybrid_rag": {
        "title": "08. HybridAI RAG",
        "subtitle": "Combined Neural Vector Retrieval + Symbolic Knowledge Graph Relational Facts",
        "module": "08_hybrid_rag",
        "badges": [{"label": "Knowledge Graph", "color": "purple"}, {"label": "Neural + Symbolic", "color": "blue"}, {"label": "Structured Reasoning", "color": "green"}],
        "default_query": "Who is the CEO of the company that owns Nimbus Cloud?",
        "slide_example": "Who is the CEO of the company that owns Instagram?",
        "how_it_works": [
            "Extracts entity names from user prompt.",
            "Performs symbolic lookup on Knowledge Graph (NetworkX DiGraph triples).",
            "Executes vector retrieval for unstructured descriptive content.",
            "Synthesizes answer using KG facts for exact relations and vector docs for details."
        ],
        "best_when": "Structured entity relationships and unstructured descriptive documents both matter.",
        "usage": "Enterprise knowledge graphs, Compliance systems, Complex relational QA.",
        "mermaid": """graph TD
    User([User Query]) --> Extract[Extract Entities]
    Extract --> KG[Knowledge Graph Lookup]
    Extract --> Vector[Vector DB Search]
    KG -->|Structured Triples| Reason[Structured Hybrid Reasoning]
    Vector -->|Unstructured Context| Reason
    Reason --> Answer([Grounded Relational Answer])
    style KG fill:#0f172a,stroke:#c084fc,color:#f8fafc
    style Vector fill:#0f172a,stroke:#38bdf8,color:#f8fafc
    style Reason fill:#1e293b,stroke:#4ade80,color:#f8fafc
    style User fill:#0f172a,stroke:#38bdf8,color:#f8fafc
    style Extract fill:#1e293b,stroke:#38bdf8,color:#f8fafc
    style Answer fill:#0f172a,stroke:#4ade80,color:#f8fafc"""
    },
    "09_cost_constrained_rag": {
        "title": "09. Cost-Constrained RAG",
        "subtitle": "Dynamic Budget Tracking, Token Estimation and Model Tier Allocation",
        "module": "09_cost_constrained_rag",
        "badges": [{"label": "Budget Tracking", "color": "amber"}, {"label": "Dynamic Models", "color": "blue"}, {"label": "Cost Optimization", "color": "green"}],
        "default_query": "Summarize the security policy and access controls in detail.",
        "slide_example": "Check budget then use smaller model and fewer chunks.",
        "how_it_works": [
            "Monitors remaining token budget before execution.",
            "High Budget: gpt-oss-120b + Top-4 Chunks. Low Budget: gpt-oss-20b + Top-2 Chunks.",
            "Executes retrieval and generation using chosen tier parameters.",
            "Calculates estimated tokens spent and decrements budget counter."
        ],
        "best_when": "Operating high-volume production SaaS applications where API costs must be bounded.",
        "usage": "High-volume SaaS chatbots, Tiered customer plans, Production APIs at scale.",
        "mermaid": """graph TD
    User([User Query]) --> BudgetCheck{Check Token Budget}
    BudgetCheck -->|High Budget| Premium[Full Tier: Large Model + 4 Chunks]
    BudgetCheck -->|Low Budget| Economy[Economy Tier: Fast Model + 2 Chunks]
    Premium --> Generate[Execute & Deduct Budget]
    Economy --> Generate
    Generate --> Answer([Cost-Aware Answer])
    style BudgetCheck fill:#0f172a,stroke:#fbbf24,color:#f8fafc
    style Premium fill:#1e293b,stroke:#38bdf8,color:#f8fafc
    style Economy fill:#1e293b,stroke:#4ade80,color:#f8fafc
    style User fill:#0f172a,stroke:#38bdf8,color:#f8fafc
    style Generate fill:#1e293b,stroke:#38bdf8,color:#f8fafc
    style Answer fill:#0f172a,stroke:#4ade80,color:#f8fafc"""
    },
    "10_xai_rag": {
        "title": "10. XAI RAG (Explainable AI)",
        "subtitle": "Auditable Answers with Step-by-Step Reasoning Traces and Cited Evidence",
        "module": "10_xai_rag",
        "badges": [{"label": "Explainable", "color": "purple"}, {"label": "Cited Sources", "color": "green"}, {"label": "Audit Trail", "color": "blue"}],
        "default_query": "Why would an enterprise application for Nimbus Cloud be rejected?",
        "slide_example": "Why was this loan rejected? Retrieves policy docs, cites rules and reasoning.",
        "how_it_works": [
            "Retrieves source documents and formats numbered context blocks.",
            "Enforces structured output schema (CitedAnswer).",
            "LLM generates answer, reasoning trace, and exact cited passage IDs.",
            "UI highlights which passages were actively used in reaching the decision."
        ],
        "best_when": "Transparency and auditability are legally or operationally mandatory.",
        "usage": "Financial underwriting, Medical diagnostics, Legal analysis, Compliance auditing.",
        "mermaid": """graph TD
    User([User Query]) --> Retrieve[Retrieve Policy Docs]
    Retrieve --> StructuredLLM[Structured XAI LLM]
    StructuredLLM --> Output[Answer + Reasoning Trace + Cited IDs]
    Output --> UI[Render Auditable Answer & Citations]
    style StructuredLLM fill:#1e293b,stroke:#c084fc,color:#f8fafc
    style Output fill:#0f172a,stroke:#38bdf8,color:#f8fafc
    style UI fill:#1e293b,stroke:#4ade80,color:#f8fafc
    style User fill:#0f172a,stroke:#38bdf8,color:#f8fafc
    style Retrieve fill:#1e293b,stroke:#38bdf8,color:#f8fafc"""
    },
}


# Request / Response models
class RunRequest(BaseModel):
    architecture: str
    query: str
    extra_params: Optional[Dict[str, Any]] = None


class RunResponse(BaseModel):
    answer: str
    elapsed: float
    documents: List[Dict[str, Any]]
    traces: Dict[str, Any]


# API Endpoints
@app.get("/api/architectures")

def list_architectures():
    """Return metadata for all 10 architectures."""
    result = {}
    for key, info in ARCHITECTURES.items():
        result[key] = {k: v for k, v in info.items() if k != "module"}
        result[key]["key"] = key
    return result


@app.post("/api/run")
def run_pipeline(req: RunRequest):
    """Execute a RAG architecture pipeline."""
    if req.architecture not in ARCHITECTURES:
        raise HTTPException(status_code=400, detail=f"Unknown architecture: {req.architecture}")

    arch = ARCHITECTURES[req.architecture]
    extra = req.extra_params or {}

    start_time = time.time()
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        mod = importlib.import_module(arch["module"])
        vdb = get_vector_db()

        # Build graph
        if req.architecture == "08_hybrid_rag":
            kg = mod.build_knowledge_graph()
            graph = mod.make_graph(vdb, kg)
        else:
            graph = mod.make_graph(vdb)

        # Prepare inputs
        if req.architecture == "01_standard_rag":
            inputs = {"question": req.query}
        elif req.architecture == "02_agentic_rag":
            inputs = {
                "messages": [
                    SystemMessage(content=mod.SYSTEM_PROMPT),
                    HumanMessage(content=req.query),
                ]
            }
        elif req.architecture == "03_memory_rag":
            inputs = {
                "messages": [HumanMessage(content=req.query)],
                "user_id": extra.get("user_id", "user-42"),
                "question": req.query,
            }
        elif req.architecture == "04_self_rag":
            inputs = {"question": req.query, "retries": 0}
        elif req.architecture == "05_adaptive_rag":
            inputs = {"question": req.query, "sub_queries": [], "documents": []}
        elif req.architecture == "06_corrective_rag":
            inputs = {"question": req.query, "original_question": req.query, "retries": 0}
        elif req.architecture == "07_attention_rag":
            inputs = {"question": req.query}
        elif req.architecture == "08_hybrid_rag":
            inputs = {"question": req.query}
        elif req.architecture == "09_cost_constrained_rag":
            inputs = {"question": req.query, "budget_remaining": extra.get("budget_remaining", 1000)}
        elif req.architecture == "10_xai_rag":
            inputs = {"question": req.query}
        else:
            inputs = {"question": req.query}

        # Invoke
        if req.architecture == "03_memory_rag":
            config = {"configurable": {"thread_id": "web-demo-thread"}}
            result = graph.invoke(inputs, config=config)
        else:
            result = graph.invoke(inputs)

        elapsed = time.time() - start_time

        # Extract answer
        if req.architecture == "02_agentic_rag":
            answer = result["messages"][-1].content
        else:
            answer = result.get("answer", "")

        # Serialize documents
        raw_docs = result.get("documents", [])
        docs = []
        for d in raw_docs:
            if hasattr(d, "page_content"):
                docs.append({"page_content": d.page_content, "metadata": d.metadata})
            else:
                docs.append({"page_content": str(d), "metadata": {}})

        # Architecture-specific traces
        traces: Dict[str, Any] = {}
        if req.architecture == "03_memory_rag":
            mem_mod = importlib.import_module("03_memory_rag")
            traces["long_term_memory"] = mem_mod.LONG_TERM_MEMORY
        elif req.architecture == "04_self_rag":
            traces["retries"] = result.get("retries", 0)
            traces["verdict"] = result.get("verdict", "N/A")
        elif req.architecture == "05_adaptive_rag":
            traces["complexity"] = result.get("complexity", "")
            traces["sub_queries"] = result.get("sub_queries", [])
        elif req.architecture == "06_corrective_rag":
            traces["relevant_ratio"] = result.get("relevant_ratio", 0.0)
            traces["retries"] = result.get("retries", 0)
        elif req.architecture == "07_attention_rag":
            weighted = result.get("weighted", [])
            traces["weighted"] = [
                {"source": d.metadata.get("source", "unknown") if hasattr(d, "metadata") else "unknown", "weight": w}
                for d, w in weighted
            ]
        elif req.architecture == "08_hybrid_rag":
            traces["entities"] = result.get("entities", [])
            traces["kg_facts"] = result.get("kg_facts", [])
        elif req.architecture == "09_cost_constrained_rag":
            traces["tier"] = result.get("tier", "")
            traces["model"] = result.get("model", "")
            traces["tokens_spent"] = result.get("tokens_spent", 0)
            traces["budget_remaining"] = result.get("budget_remaining", 0)
        elif req.architecture == "10_xai_rag":
            traces["reasoning_trace"] = result.get("reasoning_trace", "")
            traces["cited_source_ids"] = result.get("cited_source_ids", [])
        elif req.architecture == "02_agentic_rag":
            tool_results = []
            for m in result.get("messages", []):
                if getattr(m, "type", None) == "tool":
                    tool_results.append(m.content[:500])
            traces["tool_results"] = tool_results

        return {
            "answer": answer,
            "elapsed": round(elapsed, 2),
            "documents": docs,
            "traces": traces,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload custom documents to the vector DB."""
    global _custom_vectorstore
    try:
        from langchain_core.documents import Document

        common = _get_common()
        user_documents = []
        for f in files:
            content = (await f.read()).decode("utf-8", errors="ignore")
            if content.strip():
                user_documents.append(
                    Document(page_content=content, metadata={"source": f.filename, "topic": "custom_upload"})
                )

        if not user_documents:
            raise HTTPException(status_code=400, detail="No valid documents found in uploaded files.")

        coll_name = f"custom_kb_{int(time.time())}"
        _custom_vectorstore = common.build_vectorstore(docs=user_documents, collection_name=coll_name)

        return {"message": f"Indexed {len(user_documents)} document(s)", "count": len(user_documents)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reset-kb")
def reset_kb():
    """Reset to default knowledge base."""
    global _custom_vectorstore
    _custom_vectorstore = None
    return {"message": "Reset to default Nimbus Cloud knowledge base."}


@app.get("/api/health")
def health():
    return {"status": "ok", "groq_key_set": bool(os.getenv("GROQ_API_KEY"))}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
