from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings

COLLECTION_NAME = "laravel-13x-docs"
PERSIST_DIR = "laravel_docs_db"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1"


def format_docs(docs):
    parts = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "")
        section = d.metadata.get("section_path", d.metadata.get("section", "?"))
        title = d.metadata.get("title", "?")
        parts.append(f"[{i}] {section} > {title}\nURL: {source}\n{d.page_content}")
    return "\n\n".join(parts)


def build_chain():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vs = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.35}
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a Laravel docs assistant. Use ONLY the provided context. "
            "If the answer is not in context, say you don't know. "
            "Cite sources using [1], [2], ...",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])

    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    chain = (
        {
            "question": RunnablePassthrough(),
            "context": retriever | format_docs,
        }
        | prompt
        | llm | StrOutputParser()
    )

    return chain




def search(vector_store, query, k=3):
    results = vector_store.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=max(12, k * 4),
        lambda_mult=0.35,
    )
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
    chain = build_chain()
    while True:
        q = input("\nAsk: ").strip()
        if not q:
            continue

        print("\n" + chain.invoke(q))
