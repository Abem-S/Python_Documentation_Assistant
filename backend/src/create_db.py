from langchain_community.vectorstores import Chroma
# The key change: import Google's embedding model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chunk import get_chunks_from_docs
from load import load_local_docs
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PERSIST_DIRECTORY = 'chroma_db'

def create_chroma_db():
    docs_to_chunk = load_local_docs()
    chunks = get_chunks_from_docs(docs_to_chunk)
    print(f"Starting to create ChromaDB with {len(chunks)} chunks.")
    # This is the key change to use the Google API for embeddings
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIRECTORY
    )
    vector_db.persist()
    print(f"Successfully created and saved ChromaDB to '{PERSIST_DIRECTORY}'")

if __name__ == "__main__":
    create_chroma_db()
