import sys
import os

sys.path.append(os.path.abspath("."))
from utils.retriever import load_retriever

retriever = load_retriever()

docs = retriever.invoke("high engine temperature")
for i, d in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(d.page_content)