from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chunk import get_chunks_from_docs
from load import load_local_docs
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set an absolute path for the database
# This will create the database at the project's root folder,
# which is a reliable location for the app to find it later.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PERSIST_DIRECTORY = os.path.join(PROJECT_ROOT, "chroma_db")

def create_chroma_db():
    print("Starting the database creation process...")

    # Load documents and chunk them
    docs_to_chunk = load_local_docs()
    if not docs_to_chunk:
        print("ERROR: No documents found to process. Please check your data directory.")
        return
    
    chunks = get_chunks_from_docs(docs_to_chunk)
    if not chunks:
        print("ERROR: No chunks were created. Please check the chunking process.")
        return

    print(f"Successfully loaded {len(docs_to_chunk)} documents and created {len(chunks)} chunks.")
    
    # Initialize the embedding model
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        print("GoogleGenerativeAIEmbeddings model initialized successfully.")
    except Exception as e:
        print(f"ERROR: Failed to initialize embedding model: {e}")
        return

    # Create and persist the vector database
    try:
        # Check and create the directory if it doesn't exist
        if not os.path.exists(PERSIST_DIRECTORY):
            os.makedirs(PERSIST_DIRECTORY)
            print(f"Created directory: '{PERSIST_DIRECTORY}'")

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=PERSIST_DIRECTORY
        )
        vector_db.persist()
        print(f"Successfully created and saved ChromaDB to '{PERSIST_DIRECTORY}'")
    except Exception as e:
        print(f"ERROR: Failed to create or save ChromaDB: {e}")

if __name__ == "__main__":
    create_chroma_db()
