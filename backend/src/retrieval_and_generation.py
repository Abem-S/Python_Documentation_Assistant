# backend/src/retrieval_and_generation.py

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Load environment variables from the .env file in the backend directory
load_dotenv()

# Define constants
PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Check if the API key is available
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Initialize the Chroma vector store
vector_store = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_model
)

# Initialize the Groq LLM
groq_llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-8b-8192"
)

# Initialize the RAG chain. This is done once when the script loads.
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
    response = rag_chain.invoke(query)
    return response

