# backend/src/retrieval_and_generation.py

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
# The key change: import Google's embedding model
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables from the .env file in the backend directory
load_dotenv()

# Define constants
PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Check if the API key is available
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")

# Initialize the embedding model
# This is the key change to use the Google API for embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

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

