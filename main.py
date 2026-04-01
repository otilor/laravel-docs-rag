import getpass
import os
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_ollama import OllamaEmbeddings


os.environ["LANGSMITH_tracing"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()

#load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/", ),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    )
)

docs = loader.load()

embeddings = OllamaEmbeddings(model="llama3.1")


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
vector_1 = embeddings.embed_query(all_splits[0].page_content)

# print(vector_1)
# exit

vector_store= Chroma(
    embedding_function=embeddings,
    persist_directory='my_file_db',
    collection_name='sample',
)
# Index chunks
_ = vector_store.add_documents(documents=all_splits)


results = vector_store.similarity_search(
    "What does chain of hindsight mean?"
)
print(results)