from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  
from chunk import get_chunks_from_docs
from load import load_local_docs

PERSIST_DIRECTORY = 'chroma_db'

def create_chroma_db():
    docs_to_chunk = load_local_docs()
    chunks = get_chunks_from_docs(docs_to_chunk)
    print(f"Starting to create ChromaDB with {len(chunks)} chunks.")
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIRECTORY
    )
    print(f"Successfully created and saved ChromaDB to '{PERSIST_DIRECTORY}'")

if __name__ == "__main__":
    create_chroma_db()