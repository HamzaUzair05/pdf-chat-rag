# 📄💬 PDF Chat with Local RAG + Ollama

A local RAG (Retrieval-Augmented Generation) application that allows you to chat with your PDF documents using Ollama and LangChain. No API keys required - everything runs locally!

## ✨ Features

- **Upload and chat with PDF documents**
- **Local processing** - No data sent to external APIs
- **Three interfaces**: Streamlit web UI, terminal-based chat, and **MCP tools for Claude**
- **Conversation memory** - Maintains context across questions
- **Vector storage** with ChromaDB for efficient document retrieval
- **MCP Server** - Use as tools in Claude Desktop via Model Context Protocol

## 🛠️ Tech Stack

- **LangChain** - Framework for LLM applications
- **Ollama** - Local LLM inference
- **ChromaDB** - Vector database for embeddings
- **Streamlit** - Web interface
- **PyPDF** - PDF processing
- **MCP (Model Context Protocol)** - Integration with Claude Desktop

## 📋 Prerequisites

1. **Install Ollama** from [ollama.ai](https://ollama.ai)
2. **Pull required models**:
   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
   ```

##  Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/HamzaUzair05/pdf-chat-rag.git
   cd pdf-chat-rag
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv rag-env
   rag-env\Scripts\activate  # Windows
   # source rag-env/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

##  Usage

### Streamlit Web Interface
```bash
streamlit run app.py
```
- Upload a PDF file
- Start chatting with your document
- View conversation history

### Terminal Interface

1. **Process PDF first**:
   ```bash
   python rag_pdf_chat.py
   ```
   (Make sure to update the PDF filename in the script)

2. **Start chatting**:
   ```bash
   python chat_with_pdf.py
   ```

### 🔧 MCP Tools for Claude Desktop

You can now use this RAG system as MCP tools in Claude Desktop! This allows you to process PDFs and query them directly through Claude's interface.

#### Setup MCP Configuration

1. **Locate Claude Desktop config file**:
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

2. **Add the MCP server configuration**:
   ```json
   {
     "mcpServers": {
       "rag-pdf-server": {
         "command": "python",
         "args": ["mcp_server.py"],
         "cwd": "c:\\Users\\THINK\\Desktop\\rag"
       }
     }
   }
   ```

3. **Restart Claude Desktop** to load the MCP tools

#### Available MCP Tools

- **`process_pdf`** - Upload and process a PDF document into the vector database
- **`query_pdf`** - Ask questions about processed PDFs
- **`list_documents`** - View all processed documents

#### Using MCP Tools in Claude

Once configured, you can use natural language in Claude Desktop:
- "Process this PDF file for me: /path/to/document.pdf"
- "What does the document say about [topic]?"
- "Show me all the documents I've processed"

##  Project Structure

```
pdf-chat-rag/
├── app.py                 # Streamlit web interface
├── rag_pdf_chat.py       # PDF processing script
├── chat_with_pdf.py      # Terminal chat interface
├── mcp_server.py         # MCP server for Claude integration
├── mcp_config.json       # Example MCP configuration
├── requirements.txt      # Python dependencies
├── db/                   # ChromaDB vector storage
└── README.md            # This file
```

## 🔧 Configuration

- **Chunk size**: 500 characters (adjustable in code)
- **Chunk overlap**: 100 characters
- **Retrieval**: Top 3 similar chunks
- **Models**: llama3 (chat), nomic-embed-text (embeddings)

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [Ollama](https://ollama.ai/) for local LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Model Context Protocol](https://modelcontextprotocol.io/) for Claude integration

---
**Made by [Hamza Uzair](https://github.com/HamzaUzair05)**