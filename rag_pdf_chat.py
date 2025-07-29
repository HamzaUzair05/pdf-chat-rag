from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


# Load and split PDF
loader = PyPDFLoader("Uno-Flip-Manual.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

# Embed with Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")  # or try llama2 embeddings
db = Chroma.from_documents(chunks, embeddings, persist_directory="./db")
db.persist()
