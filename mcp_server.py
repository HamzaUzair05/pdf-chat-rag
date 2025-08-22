import asyncio
import os
import sys
from typing import Optional, List, Dict, Any
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
import mcp.server.stdio
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Global storage
sessions: Dict[str, ConversationalRetrievalChain] = {}
vector_store: Optional[Chroma] = None
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3")

# Create MCP server
server = Server("rag-pdf-server")

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """List available RAG resources."""
    resources = []
    
    if vector_store:
        resources.append(
            types.Resource(
                uri=AnyUrl("rag://pdf-knowledge-base"),
                name="PDF Knowledge Base",
                description="RAG-enabled PDF knowledge base for Q&A",
                mimeType="application/json",
            )
        )
    
    # List active sessions
    for session_id in sessions.keys():
        resources.append(
            types.Resource(
                uri=AnyUrl(f"rag://session/{session_id}"),
                name=f"Chat Session: {session_id}",
                description=f"Conversation session {session_id}",
                mimeType="application/json",
            )
        )
    
    return resources

@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    """Read resource content."""
    uri_str = str(uri)
    
    if uri_str == "rag://pdf-knowledge-base":
        if not vector_store:
            return "No PDF knowledge base loaded"
        
        # Get some sample content from vector store
        try:
            docs = vector_store.similarity_search("", k=3)
            content = {
                "status": "loaded",
                "documents_count": len(docs),
                "sample_content": [doc.page_content[:200] for doc in docs]
            }
            return str(content)
        except:
            return "Knowledge base loaded but no content available"
    
    elif uri_str.startswith("rag://session/"):
        session_id = uri_str.split("/")[-1]
        if session_id in sessions:
            return f"Active chat session: {session_id}"
        else:
            return f"Session {session_id} not found"
    
    raise ValueError(f"Unknown resource: {uri}")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name="process_pdf",
            description="Process a PDF file and create vector embeddings for RAG",
            inputSchema={
                "type": "object",
                "properties": {
                    "pdf_path": {
                        "type": "string",
                        "description": "Path to the PDF file"
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "Size of text chunks",
                        "default": 500
                    },
                    "chunk_overlap": {
                        "type": "integer", 
                        "description": "Overlap between chunks",
                        "default": 100
                    }
                },
                "required": ["pdf_path"]
            },
        ),
        types.Tool(
            name="query_rag",
            description="Query the RAG system with a question",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Question to ask the RAG system"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session ID for conversation memory",
                        "default": "default"
                    }
                },
                "required": ["question"]
            },
        ),
        types.Tool(
            name="load_existing_vectorstore",
            description="Load existing vector store from disk",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        types.Tool(
            name="list_sessions",
            description="List all active chat sessions",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls."""
    global vector_store, sessions
    
    if name == "process_pdf":
        pdf_path = arguments["pdf_path"]
        chunk_size = arguments.get("chunk_size", 500)
        chunk_overlap = arguments.get("chunk_overlap", 100)
        
        try:
            # Load and split PDF
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_documents(docs)
            
            # Create vector store
            vector_store = Chroma.from_documents(
                chunks,
                embeddings,
                persist_directory="./db"
            )
            
            return [
                types.TextContent(
                    type="text",
                    text=f"✅ Successfully processed PDF: {len(chunks)} chunks created"
                )
            ]
        except Exception as e:
            return [
                types.TextContent(
                    type="text", 
                    text=f"❌ Error processing PDF: {str(e)}"
                )
            ]
    
    elif name == "query_rag":
        question = arguments["question"]
        session_id = arguments.get("session_id", "default")
        
        if not vector_store:
            return [
                types.TextContent(
                    type="text",
                    text="❌ No PDF processed yet. Please process a PDF first."
                )
            ]
        
        try:
            # Get or create session
            if session_id not in sessions:
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                
                sessions[session_id] = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    memory=memory
                )
            
            # Query the chain
            qa_chain = sessions[session_id]
            result = qa_chain.invoke({"question": question})
            
            answer = result["answer"]
            
            # Add source info if available
            if "source_documents" in result:
                sources = "\n\n📚 Sources:\n" + "\n".join([
                    f"- {doc.page_content[:100]}..." 
                    for doc in result["source_documents"][:2]
                ])
                answer += sources
            
            return [
                types.TextContent(
                    type="text",
                    text=answer
                )
            ]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=f"❌ Error querying RAG: {str(e)}"
                )
            ]
    
    elif name == "load_existing_vectorstore":
        try:
            if os.path.exists("./db"):
                vector_store = Chroma(persist_directory="./db", embedding_function=embeddings)
                return [
                    types.TextContent(
                        type="text",
                        text="✅ Existing vector store loaded successfully"
                    )
                ]
            else:
                return [
                    types.TextContent(
                        type="text", 
                        text="❌ No existing vector store found at ./db"
                    )
                ]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=f"❌ Error loading vector store: {str(e)}"
                )
            ]
    
    elif name == "list_sessions":
        session_list = list(sessions.keys()) if sessions else ["No active sessions"]
        return [
            types.TextContent(
                type="text",
                text=f"Active sessions: {', '.join(session_list)}"
            )
        ]
    
    raise ValueError(f"Unknown tool: {name}")

async def main():
    global vector_store  # Add this line!
    
    # Load existing vector store if available
    try:
        if os.path.exists("./db"):
            vector_store = Chroma(persist_directory="./db", embedding_function=embeddings)
            print("Loaded existing vector store", file=sys.stderr)
    except Exception as e:
        print(f"Could not load existing vector store: {e}", file=sys.stderr)
    
    # Run the server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="rag-pdf-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
