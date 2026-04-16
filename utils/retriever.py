import os
from functools import lru_cache
from typing import List

from langchain_core.documents import Document


class KeywordRetriever:
    def __init__(self, sections: List[str], k: int = 3):
        self.sections = sections
        self.k = k

    def invoke(self, query: str) -> List[Document]:
        terms = [term.lower() for term in query.split() if term.strip()]
        scored_sections = []

        for section in self.sections:
            section_lower = section.lower()
            score = sum(section_lower.count(term) for term in terms)
            scored_sections.append((score, section))

        ranked = sorted(scored_sections, key=lambda item: item[0], reverse=True)
        top_sections = [section for score, section in ranked[: self.k] if score > 0]

        if not top_sections:
            top_sections = self.sections[: self.k]

        return [Document(page_content=section) for section in top_sections]


@lru_cache(maxsize=1)
def load_retriever():
    file_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "maintenance_docs.txt",
    )

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    sections = [section.strip() for section in text.split("---") if section.strip()]

    use_vector_retriever = os.getenv("USE_VECTOR_RETRIEVER", "0") == "1"

    if use_vector_retriever:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"local_files_only": True},
        )
        vectorstore = FAISS.from_texts(sections, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 3})

    return KeywordRetriever(sections, k=3)
