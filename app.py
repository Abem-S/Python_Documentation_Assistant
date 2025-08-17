import streamlit as st
from retrieval_and_generation import rag_chain

st.set_page_config(
    page_title="Python Docs Assistant",
    page_icon="🐍",
    layout="wide"
)

# CACHED RESOURCES
@st.cache_resource
def get_rag_chain():
    """Load and return the RAG chain."""
    return rag_chain

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

    # Invoke the RAG chain
    rag_response = rag_chain.invoke(query)
    answer = rag_response.get("result", "I'm sorry, I couldn't find an answer.")
    sources = rag_response.get("source_documents", [])

    # Format sources for display
    source_links = []
    seen_urls = set()
    for doc in sources:
        url = doc.metadata.get("source") or doc.metadata.get("url")
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

    # Clear button to reset chat history
    def clear_chat():
        st.session_state.messages = []

    st.button("Clear Chat", on_click=clear_chat)
