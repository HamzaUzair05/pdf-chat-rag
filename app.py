import os
import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma as ChromaStore

st.set_page_config(page_title="Chat with your PDF", layout="wide")
st.title("📄💬 Chat with your PDF (Local RAG + Ollama)")

# Initialize session state for memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):

        # Save PDF locally
        pdf_path = os.path.join("temp.pdf")
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load and split PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        # Create embeddings
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Store in Chroma
        # Create or load vector store properly
        db = ChromaStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./db"
        )
        # Build retrieval chain
        retriever = db.as_retriever(search_kwargs={"k": 3})
        llm = ChatOllama(model="llama3")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )

        st.success("PDF processed. Start chatting below!")

# Chat interface
if st.session_state.qa_chain:
    user_input = st.text_input("Ask something about the PDF:")
    if user_input:
        response = st.session_state.qa_chain.invoke({"question": user_input})
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

# Display chat history
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")

# Clear chat history
if st.button("🧹 Clear chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()
