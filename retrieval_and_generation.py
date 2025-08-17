import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
PERSIST_DIRECTORY = "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

vector_store = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_model
)

groq_llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-8b-8192"
)

rag_chain = RetrievalQA.from_chain_type(
    llm=groq_llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)



