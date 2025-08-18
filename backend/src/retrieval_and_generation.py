import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables from the .env file in the backend directory
load_dotenv()

# Define constants
# Use the same absolute path as the create_db.py script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PERSIST_DIRECTORY = os.path.join(PROJECT_ROOT, "chroma_db")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Check if the API key is available
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")

# Initialize the embedding model
try:
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    print("Embedding model initialized for retrieval.")
except Exception as e:
    print(f"ERROR: Failed to initialize embedding model for retrieval: {e}")
    # Re-raise the exception to stop the application from running in a broken state
    raise

# Initialize the Chroma vector store
try:
    print(f"Attempting to load Chroma vector store from: {PERSIST_DIRECTORY}")
    if not os.path.exists(PERSIST_DIRECTORY):
        print("ERROR: ChromaDB directory not found. Please check your Render build logs.")
        raise FileNotFoundError("ChromaDB directory not found.")

    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model
    )
    print("Chroma vector store loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load Chroma vector store: {e}")
    # Re-raise the exception to stop the application from running in a broken state
    raise

# Initialize the Groq LLM
groq_llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-8b-8192"
)

# Initialize the RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=groq_llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

def get_rag_response(query: str):
    """
    Runs a query through the pre-initialized RAG chain.

    Args:
        query (str): The user's question or prompt.

    Returns:
        A dictionary containing the RAG chain's result and source documents.
    """
    response = rag_chain.invoke({"query": query})
    return response
