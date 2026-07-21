"""
common.py
=========
Shared setup used by every `NN_*_rag.py` script in this folder:

- Groq LLM (openai/gpt-oss-20b or openai/gpt-oss-120b) via `langchain_groq`
- HuggingFace embeddings (any sentence-transformers model) via `langchain_huggingface`
- An in-memory Chroma vector store, pre-loaded with a small demo knowledge base
- A couple of small utilities used across scripts

Each architecture file imports what it needs from here so the interesting
part of every file is just the LangGraph graph itself.

Environment variables (put these in a `.env` file next to this one):
    GROQ_API_KEY     required   -> https://console.groq.com/keys
    GROQ_MODEL       optional   default: openai/gpt-oss-20b
                                 (swap to openai/gpt-oss-120b for a bigger model)
    HF_EMBED_MODEL   optional   default: sentence-transformers/all-MiniLM-L6-v2
                                 (any HF sentence-transformers model works)
"""

from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from typing import List, Optional



from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_MODEL_LARGE = os.getenv("GROQ_MODEL_LARGE", "openai/gpt-oss-120b")
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


# LLM and embeddings factories

def get_llm(temperature: float = 0.2, model: Optional[str] = None, api_key: Optional[str] = None) -> ChatGroq:
    """Return a Groq chat model. Pass model=GROQ_MODEL_LARGE for gpt-oss-120b."""
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to a .env file or export it in your shell."
        )
    model_name = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    return ChatGroq(model=model_name, temperature=temperature, groq_api_key=key)



_embeddings_cache: dict[str, HuggingFaceEmbeddings] = {}


def get_embeddings(model_name: Optional[str] = None):
    """Return (and cache) embedding model with instant local lookup and fallback."""
    name = model_name or HF_EMBED_MODEL
    if name not in _embeddings_cache:
        try:
            _embeddings_cache[name] = HuggingFaceEmbeddings(
                model_name=name,
                model_kwargs={"device": "cpu"},
                multi_process=False
            )
        except Exception:
            from langchain_community.embeddings import DeterministicFakeEmbedding
            _embeddings_cache[name] = DeterministicFakeEmbedding(size=384)
    return _embeddings_cache[name]




# Demo knowledge base

SAMPLE_DOCS: List[Document] = [
    Document(
        page_content=(
            "Nimbus Cloud is a managed object-storage platform. It offers three tiers: "
            "Starter (100GB, $9/mo), Team (2TB, $49/mo), and Enterprise (custom capacity, "
            "custom pricing). All tiers include automatic versioning and daily backups."
        ),
        metadata={"id": "pricing-01", "topic": "pricing", "source": "pricing_guide.md"},
    ),
    Document(
        page_content=(
            "Nimbus Cloud revenue grew from $18M to $27M year-over-year, a 50% increase, "
            "driven mainly by Enterprise tier upgrades. Customer churn dropped from 6% to 4%."
        ),
        metadata={"id": "finance-01", "topic": "finance", "source": "annual_report.md"},
    ),
    Document(
        page_content=(
            "Nimbus Cloud is SOC 2 Type II certified. All data is encrypted at rest with "
            "AES-256 and in transit with TLS 1.3. Access control is role-based (RBAC) and "
            "every action is written to an immutable audit log."
        ),
        metadata={"id": "security-01", "topic": "security", "source": "security_policy.md"},
    ),
    Document(
        page_content=(
            "Refunds are granted within 14 days of purchase if usage is under 5GB. "
            "Enterprise contracts follow the terms negotiated in the signed agreement, "
            "not the standard refund window. Refund requests must go through support."
        ),
        metadata={"id": "policy-01", "topic": "policy", "source": "refund_policy.md"},
    ),
    Document(
        page_content=(
            "Loan-style credit checks are not part of Nimbus Cloud's product. Enterprise "
            "applications are instead reviewed against a usage-risk rubric: account age, "
            "payment history, and requested capacity. Applications are rejected if payment "
            "history shows more than two failed charges in the last 90 days."
        ),
        metadata={"id": "policy-02", "topic": "policy", "source": "enterprise_review.md"},
    ),
    Document(
        page_content=(
            "Onboarding steps: 1) create an organization, 2) invite teammates by email, "
            "3) create a bucket and set its region, 4) generate an API key, 5) run the "
            "quick-start CLI command to upload a test file."
        ),
        metadata={"id": "howto-01", "topic": "onboarding", "source": "onboarding_guide.md"},
    ),
    Document(
        page_content=(
            "Common upload errors: Err-401 means the API key is invalid or expired. "
            "Err-413 means the file exceeds the 5GB single-upload limit and should use "
            "multipart upload instead. Err-503 is a transient region outage; retry with backoff."
        ),
        metadata={"id": "howto-02", "topic": "troubleshooting", "source": "error_codes.md"},
    ),
    Document(
        page_content=(
            "Nimbus Cloud is a subsidiary of Orbit Holdings. Orbit Holdings also owns "
            "Beacon Analytics and Drift Messaging. Orbit Holdings' CEO is Maria Ferreira, "
            "who founded the parent company in 2015."
        ),
        metadata={"id": "corp-01", "topic": "corporate", "source": "company_structure.md"},
    ),
    Document(
        page_content=(
            "Nimbus Cloud supports vegetarian-friendly office catering at its partner "
            "sites for on-site training days: catering vendors are required to offer at "
            "least two high-protein vegetarian options (lentil, tofu, or legume based)."
        ),
        metadata={"id": "misc-01", "topic": "misc", "source": "vendor_requirements.md"},
    ),
    Document(
        page_content=(
            "Multipart upload works by splitting a file into parts of at least 5MB each "
            "(except the last part), uploading each part with its own checksum, then "
            "calling complete-multipart-upload with the ordered list of part ETags."
        ),
        metadata={"id": "howto-03", "topic": "troubleshooting", "source": "multipart_guide.md"},
    ),
]


_vectorstore_cache: dict[str, Chroma] = {}


def build_vectorstore(
    docs: Optional[List[Document]] = None,
    collection_name: str = "nimbus_demo",
    chunk_size: int = 400,
    chunk_overlap: int = 40,
) -> Chroma:
    """Split docs, embed them, and load them into an in-memory Chroma collection."""
    if docs is None and collection_name in _vectorstore_cache:
        return _vectorstore_cache[collection_name]

    docs = docs if docs is not None else SAMPLE_DOCS
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    vdb = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=collection_name,
    )
    if docs is None or collection_name == "nimbus_demo":
        _vectorstore_cache[collection_name] = vdb
    return vdb



def format_docs(docs: List[Document]) -> str:
    """Render retrieved documents as a numbered context block for prompts."""
    return "\n\n".join(
        f"[{i + 1}] (source: {d.metadata.get('source', 'unknown')}) {d.page_content}"
        for i, d in enumerate(docs)
    )
