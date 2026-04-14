import sys
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

COLLECTION_NAME = "laravel-13x-docs"
PERSIST_DIR = "laravel_docs_db"
EMBEDDING_MODEL = "nomic-embed-text"


def get_vector_store():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )


def search(vector_store, query, k=3):
    results = vector_store.similarity_search(query, k=k)
    print()
    for i, result in enumerate(results, 1):
        section_path = result.metadata.get("section_path", result.metadata.get("section", "?"))
        title = result.metadata.get("title", "?")
        source = result.metadata.get("source", "")
        print(f"--- Result {i} [{section_path} > {title}] ---")
        print(f"URL: {source}")
        print(result.page_content[:500])
        print()


if __name__ == "__main__":
    vector_store = get_vector_store()
    count = vector_store._collection.count()

    if count == 0:
        print("No documents indexed yet. Gotta index the docs first.")
        sys.exit(1)

    print(f"Laravel 13.x docs index loaded ({count} chunks).")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        search(vector_store, query)
    else:
        while True:
            try:
                query = input("\nAsk a question (ctrl+c to quit): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break
            if not query:
                continue
            search(vector_store, query)
