from langchain.text_splitter import RecursiveCharacterTextSplitter
from load import load_local_docs 

def get_chunks_from_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    return chunks

if __name__ == "__main__":
    docs_to_chunk = load_local_docs()
    chunks = get_chunks_from_docs(docs_to_chunk)
    
    if chunks:
        print("Source:", chunks[0].metadata.get('source'))
        print("Content chunked:", chunks[0].page_content[:500])