from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory

# Initialize embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load vector DB
db = Chroma(persist_directory="./db", embedding_function=embeddings)

# Build retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# Load local LLM
llm = ChatOllama(model="llama3")

# Memory for conversation context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create RAG chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# Chat loop
print("Ask questions about your PDF (type 'exit' to quit):")
while True:
    query = input("\nYou: ")
    if query.lower() in ['exit', 'quit']:
        break

    result = qa_chain.invoke({"question": query})
    print("Bot:", result)
