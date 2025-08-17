from langchain_community.document_loaders import DirectoryLoader, TextLoader

def load_local_docs():
    folder_name = 'python-3.13-docs-text'
    loader = DirectoryLoader(
        folder_name,
        loader_cls=TextLoader,
        recursive=True
    )
    docs = loader.load()
    return docs

if __name__ == "__main__":
    docs = load_local_docs()
    print(f"Successfully loaded {len(docs)} documents.")
