import json
import requests
import bs4
from langchain_core.documents import Document
from langchain_text_splitters import HTMLSemanticPreservingSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

BASE_URL = "https://laravel.com"
DOCS_URL = f"{BASE_URL}/docs/13.x"
COLLECTION_NAME = "laravel-13x-docs"
PERSIST_DIR = "laravel_docs_db"
EMBEDDING_MODEL = "nomic-embed-text"
MAX_EMBED_CHARS = 6000


def fetch_doc_urls():
    """Extract all documentation page URLs from the embedded Inertia JSON."""
    response = requests.get(DOCS_URL)
    response.raise_for_status()

    soup = bs4.BeautifulSoup(response.text, "html.parser")
    script_tag = soup.find("script", {"data-page": "app", "type": "application/json"})
    page_data = json.loads(script_tag.string)

    urls = []
    for section in page_data["props"]["index"]:
        section_title = section["title"]
        for item in section.get("items", []):
            url = BASE_URL + item["href"]
            urls.append({
                "url": url,
                "title": item["title"],
                "section": section_title,
            })
            print(f"  [{section_title}] {item['title']} -> {url}")

    print(f"\nFound {len(urls)} documentation pages.\n")
    return urls


def load_docs(doc_urls):
    """Load docs while preserving semantic HTML for section-aware splitting."""
    docs = []
    print("Loading documentation pages...")
    for i, item in enumerate(doc_urls, 1):
        response = requests.get(item["url"])
        response.raise_for_status()
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        main_content = soup.find(id="main-content")
        if not main_content:
            print(f"  Skipping page with missing #main-content: {item['url']}")
            continue

        docs.append(
            Document(
                page_content=str(main_content), 
                metadata={
                    "source": item["url"],
                    "section": item["section"],
                    "title": item["title"],
                },
            )
        )
        if i % 25 == 0 or i == len(doc_urls):
            print(f"  Loaded {i}/{len(doc_urls)} pages...")

    print(f"Loaded {len(docs)} pages.\n")
    return docs


def split_and_index(docs):
    """Split documents into chunks and index them in Chroma."""
    text_splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=[("h1", "h1"), ("h2", "h2"), ("h3", "h3")],
        max_chunk_size=1500,
        chunk_overlap=200,
        preserve_links=True,
        elements_to_preserve=["pre", "code", "table", "ul", "ol", "li"]
    )
    overflow_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2200,
        chunk_overlap=250,
    )

    all_splits = []
    for idx, doc in enumerate(docs, 1):
        html = doc.page_content
        chunks = text_splitter.split_text(html)

        for c in chunks:
            c.metadata.update(doc.metadata)

            h1 = c.metadata.get("h1")
            h2 = c.metadata.get("h2")
            h3 = c.metadata.get("h3")
            section_path = " > ".join([x for x in [h1, h2, h3] if x])


            if section_path:
                c.metadata["section_path"] = section_path

            # HTML preserving split can still emit very large chunks for some code/table blocks.
            if len(c.page_content) > MAX_EMBED_CHARS:
                sub_chunks = overflow_splitter.split_text(c.page_content)
                for sub_chunk in sub_chunks:
                    all_splits.append(
                        Document(
                            page_content=sub_chunk,
                            metadata={**c.metadata, "overflow_split": True},
                        )
                    )
            else:
                all_splits.append(c)
        if idx % 25 == 0 or idx == len(docs):
            print(f"  Split {idx}/{len(docs)} pages...")
    print(f"Split into {len(all_splits)} chunks.\n")

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )

    existing = vector_store._collection.count()
    if existing > 0:
        print(f"Collection already has {existing} documents. Deleting and recreating...")
        vector_store.delete_collection()
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION_NAME,
        )

    batch_size = 50
    for i in range(0, len(all_splits), batch_size):
        batch = all_splits[i : i + batch_size]
        vector_store.add_documents(documents=batch)
        print(f"  Indexed batch {i // batch_size + 1} ({len(batch)} chunks)")

    print(f"\nIndexing complete. Total chunks: {vector_store._collection.count()}\n")
    return vector_store


def search(vector_store, query, k=3):
    """Run a similarity search and print results."""
    print(f'Searching: "{query}"\n')
    results = vector_store.similarity_search(query, k=k)
    for i, result in enumerate(results, 1):
        section = result.metadata.get("section", "?")
        title = result.metadata.get("title", "?")
        print(f"--- Result {i} [{section} > {title}] ---")
        print(result.page_content[:300])
        print()


if __name__ == "__main__":
    print("Fetching documentation index...\n")
    doc_urls = fetch_doc_urls()

    docs = load_docs(doc_urls)
    split_and_index(docs)
