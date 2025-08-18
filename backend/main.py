# backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# We will put the core RAG logic into a function in retrieval_and_generation.py
# and import it here.
from src.retrieval_and_generation import get_rag_response

# Load environment variables
load_dotenv()

# Initialize FastAPI application
app = FastAPI(
    title="RAG Backend API",
    description="An API for a Retrieval-Augmented Generation system.",
)

# Pydantic model to define the structure of the incoming request
# This ensures that the frontend sends a 'query' field which is a string.
class QueryRequest(BaseModel):
    query: str

# API endpoint for the RAG query
@app.post("/query")
def rag_query(request: QueryRequest):
    """
    Handles a user query by passing it to the RAG system and returning the response.
    """
    try:
        # Get the response from the RAG chain.
        # This function will be defined in the next step.
        response = get_rag_response(request.query)
        
        # Return the generated text and source documents from the response.
        # Note: The response object from your chain has specific attributes.
        # We access them here to format the response for the frontend.
        return {
            "response": response['result'],
            "source_documents": response['source_documents']
        }
        
    except Exception as e:
        # If anything goes wrong, raise an HTTP error with a descriptive message.
        raise HTTPException(status_code=500, detail=str(e))

