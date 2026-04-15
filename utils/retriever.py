import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_text_splitters import CharacterTextSplitter


def load_retriever():
    # File path
    file_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "maintenance_docs.txt"
    )

    # Read file
    with open(file_path, "r") as f:
        text = f.read()

    print("TEXT LENGTH:", len(text))

    if len(text.strip()) == 0:
        raise ValueError("File is empty")

    # 🔥 STEP 1: Split into logical sections
    sections = text.split("---")
    sections = [s.strip() for s in sections if s.strip()]

    print("SECTIONS:", len(sections))

    # 🔥 STEP 2: Chunking
    splitter = CharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )

    chunks = []
    for section in sections:
        chunks.extend(splitter.split_text(section))

    print("FINAL CHUNKS:", len(chunks))

    if len(chunks) == 0:
        raise ValueError("No valid chunks created")

    # 🔥 STEP 3: FAST EMBEDDINGS (NO FREEZE)
    embeddings = FakeEmbeddings(size=384)

    print("CREATING VECTORSTORE...")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    print("VECTORSTORE CREATED")

    # 🔥 STEP 4: Retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    return retriever