# Python Documentation Assistant

This project provides a modular pipeline for document processing, database creation, and retrieval-augmented generation using LangChain.

## Live Demo
Visit the live site: [python3-documentation-assistant.streamlit.app](https://python3-documentation-assistant.streamlit.app/)

## Folder Structure

```text
Python_Documentation_Assistant/
├── app.py                           # Main entry point for the application
├── src/
│   ├── python-3.13-docs-text/       # Local folder containing Python documentation
│   ├── .env.example                 # Example environment variables file
│   ├── chunk.py                     # Handles document chunking
│   ├── create_db.py                 # Creates a vector database from chunked documents
│   ├── load.py                      # Loads documents from various sources
│   └── retrieval_and_generation.py  # Retrieval-augmented generation
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

## File Overview

### `app.py`
Main entry point for the application. Orchestrates the workflow by integrating document chunking, database creation, loading, and retrieval/generation functionalities.

### `src/chunk.py`
Handles document chunking. Splits large documents into manageable pieces for efficient processing and embedding.

### `src/create_db.py`
Creates a vector database from chunked documents. Embeds document chunks and stores them for fast similarity search and retrieval.

### `src/load.py`
Loads documents from various sources (local files, URLs, etc.) and prepares them for chunking and embedding.

### `src/retrieval_and_generation.py`
Implements retrieval-augmented generation. Retrieves relevant document chunks from the database and generates responses using a language model.

## Usage

1. **Load Documents:** Use `src/load.py` to ingest documents.
2. **Chunk Documents:** Run `src/chunk.py` to split documents into chunks.
3. **Create Database:** Execute `src/create_db.py` to embed and store chunks.
4. **Retrieve & Generate:** Use `src/retrieval_and_generation.py` to answer queries based on retrieved chunks.
5. **Run Application:** Start the workflow with `app.py`.

## Requirements

- Python
- LangChain
- Other dependencies as specified in `requirements.txt`

