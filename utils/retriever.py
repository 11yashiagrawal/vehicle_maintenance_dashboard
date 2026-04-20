import os
import logging
from functools import lru_cache
from typing import List

from langchain_core.documents import Document


logger = logging.getLogger(__name__)

DEFAULT_MAINTENANCE_SECTIONS = [
    "Preventive maintenance baseline: check oil quality, cooling health, and brake condition every service cycle.",
    "Fault management baseline: prioritize recurring fault codes and verify subsystem stability before clearing warnings.",
    "Safety baseline: when overheating, severe vibration, or electrical instability is observed, reduce duty and inspect immediately.",
]


class KeywordRetriever:
    """
    Fallback retriever using TF-style term frequency scoring.
    Used when FAISS/sentence-transformers are unavailable.
    """
    def __init__(self, sections: List[str], k: int = 3):
        self.sections = sections
        self.k = k

    def invoke(self, query: str) -> List[Document]:
        terms = [term.lower() for term in query.split() if len(term) > 2]
        scored = []

        for i, section in enumerate(self.sections):
            section_lower = section.lower()
            score = sum(section_lower.count(term) for term in terms)
            # Boost score slightly for sections whose heading matches query terms
            first_line = section_lower.split("\n")[0]
            score += sum(2 for term in terms if term in first_line)
            scored.append((score, i, section))

        ranked = sorted(scored, key=lambda x: x[0], reverse=True)
        top = [section for score, _, section in ranked[: self.k] if score > 0]

        if not top:
            top = self.sections[: self.k]

        return [
            Document(page_content=section, metadata={"source": "maintenance_docs.txt", "retriever": "keyword"})
            for section in top
        ]


def _load_sections(file_path: str) -> List[str]:
    """Load and parse sections from maintenance_docs.txt."""
    if not os.path.exists(file_path):
        logger.warning(
            "maintenance_docs.txt not found at %s — using built-in fallback guidance.", file_path
        )
        return DEFAULT_MAINTENANCE_SECTIONS

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    sections = [s.strip() for s in text.split("---") if s.strip()]
    if not sections:
        logger.warning("maintenance_docs.txt is empty — using built-in fallback guidance.")
        return DEFAULT_MAINTENANCE_SECTIONS

    logger.info("Loaded %d sections from maintenance_docs.txt", len(sections))
    return sections


def _build_faiss_retriever(sections: List[str], k: int = 3):
    """
    Build a FAISS vector retriever using sentence-transformers.
    Downloads the model on first run (~22 MB), then caches to disk.
    Returns None if dependencies are missing or download fails.
    """
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as e:
        logger.warning("FAISS dependencies not installed (%s) — falling back to keyword retriever.", e)
        return None

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # Build documents with metadata so sources are traceable in the UI
    docs = [
        Document(
            page_content=section,
            metadata={"source": "maintenance_docs.txt", "section_index": i, "retriever": "faiss"},
        )
        for i, section in enumerate(sections)
    ]

    try:
        # Try loading cached FAISS index first (avoids re-embedding on every restart)
        cache_dir = os.path.join(os.path.dirname(__file__), "..", ".faiss_cache")
        index_path = os.path.join(cache_dir, "maintenance_index")

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            # Removed local_files_only=True — was silently failing when model not pre-downloaded
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        if os.path.exists(index_path):
            try:
                vectorstore = FAISS.load_local(
                    index_path, embeddings, allow_dangerous_deserialization=True
                )
                logger.info("Loaded cached FAISS index from %s", index_path)
                return vectorstore.as_retriever(search_kwargs={"k": k})
            except Exception as cache_exc:
                logger.warning("Cached FAISS index load failed (%s) — rebuilding.", cache_exc)

        # Build fresh index from sections
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Persist to disk so subsequent restarts are instant
        os.makedirs(cache_dir, exist_ok=True)
        vectorstore.save_local(index_path)
        logger.info("Built and cached FAISS index at %s", index_path)

        return vectorstore.as_retriever(search_kwargs={"k": k})

    except Exception as exc:
        logger.warning(
            "FAISS retriever init failed (%s) — falling back to keyword retriever.", exc
        )
        return None


@lru_cache(maxsize=1)
def load_retriever():
    """
    Load the best available retriever.

    Priority:
    1. FAISS + sentence-transformers (semantic search) — always attempted first.
    2. KeywordRetriever (TF scoring) — fallback if FAISS fails.

    The USE_VECTOR_RETRIEVER env var is now ignored — FAISS is always preferred
    and keyword is the graceful fallback. This matches the project outline requirement
    for "Chroma / FAISS (RAG)".
    """
    file_path = os.path.join(os.path.dirname(__file__), "..", "data", "maintenance_docs.txt")
    sections = _load_sections(file_path)

    faiss_retriever = _build_faiss_retriever(sections, k=3)
    if faiss_retriever is not None:
        return faiss_retriever

    logger.info("Using keyword retriever as fallback.")
    return KeywordRetriever(sections, k=3)


def get_retriever_mode() -> str:
    """
    Returns which retriever is active. Checks by attempting a cheap probe.
    Safe to call at any time — does not trigger a build.
    """
    # Check if FAISS cache exists as a proxy for whether FAISS is active
    cache_dir = os.path.join(os.path.dirname(__file__), "..", ".faiss_cache", "maintenance_index")
    try:
        from langchain_community.vectorstores import FAISS  # noqa: F401
        from langchain_huggingface import HuggingFaceEmbeddings  # noqa: F401
        return "FAISS"
    except ImportError:
        return "KEYWORD"
