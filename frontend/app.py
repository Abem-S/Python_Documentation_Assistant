# frontend/app.py
import streamlit as st
import requests  # Import the requests library
import os
from dotenv import load_dotenv

# We no longer need to import these, as the backend handles them.
# from src.retrieval_and_generation import rag_chain

# Load environment variables (for the frontend, this would be for the backend URL)
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# CONFIG
st.set_page_config(
    page_title="Python Docs Assistant",
    page_icon="🐍",
    layout="wide"
)

# SESSION STATE MANAGEMENT
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# HEADER / INTRO
st.title("🐍 Python Docs Assistant")
st.markdown(
    "An always‑on companion grounded in official Python documentation. "
    "Ask how to use features, debug workflows, or compose scripts — I’ll fetch "
    "the most relevant answers with verifiable sources."
)

# MAIN APP LOGIC
def display_message(role, content):
    """Displays a message in the chat."""
    with st.chat_message(role):
        st.markdown(content)

# Display existing messages from chat history
for message in st.session_state.messages:
    display_message(message["role"], message["content"])
    if message["role"] == "assistant" and "sources" in message:
        with st.expander("📄 Sources"):
            for source in message["sources"]:
                st.markdown(source, unsafe_allow_html=True)

# Handle user input
if query := st.chat_input("Ask me something about Python..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": query})
    display_message("user", query)
    
    # Use a loading spinner while waiting for the backend
    with st.spinner("Generating response..."):
        try:
            # Make a POST request to the backend API
            response = requests.post(
                f"{BACKEND_URL}/query",
                json={"query": query}
            )
            
            # Raise an exception for bad status codes
            response.raise_for_status()

            # Get the JSON data from the response
            backend_response = response.json()
            answer = backend_response.get("response", "I'm sorry, I couldn't find an answer.")
            sources = backend_response.get("source_documents", [])
            
            # Format sources for display
            source_links = []
            seen_urls = set()
            for doc in sources:
                url = doc.get("metadata", {}).get("source") or doc.get("metadata", {}).get("url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    # Create a clickable markdown link
                    source_links.append(f"<a href='{url}' target='_blank'>{url}</a>")

            # Display assistant's message
            display_message("assistant", answer)
            if source_links:
                with st.expander("📄 Sources"):
                    st.markdown("<br>".join(source_links), unsafe_allow_html=True)

            # Update session state with the new message and sources
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": source_links
            })
            
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to backend: {e}")
            st.info("Please make sure your backend server is running.")
            st.session_state.messages.append({"role": "assistant", "content": f"Error connecting to backend: {e}"})

    # Clear button to reset chat history
    def clear_chat():
        st.session_state.messages = []

    st.button("Clear Chat", on_click=clear_chat)
