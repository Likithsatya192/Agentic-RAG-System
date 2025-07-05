# Agentic RAG System

## Overview
Agentic RAG System is an advanced Retrieval-Augmented Generation (RAG) platform that combines local document retrieval with web search and agentic workflows. It allows users to upload documents (PDF, DOCX, TXT), ask questions, and receive synthesized answers using both local knowledge and real-time web data. The system features a modern web UI and a modular backend built with LangChain, LangGraph, and Flask.

---

## Features
- **Document Upload**: Supports PDF, DOCX, and TXT files.
- **Hybrid RAG**: Answers are generated using both local documents and web search (Tavily).
- **Agentic Workflow**: Research, analysis, and writing are handled by specialized agents.
- **Web UI**: User-friendly interface for uploading files and querying.
- **CLI Support**: Can be run and tested from the command line.
- **Extensible**: Modular codebase for easy customization and extension.

---

## Architecture
```
User (Web UI/CLI)
   |
Flask Backend (app.py)
   |
AgenticRAGSystem (main.py)
   |-- DocumentLoader (document_loader.py)
   |-- RAG & Web Tools (rag_tools.py)
   |-- Agents (agents.py)
   |-- Workflow (supervised_workflow.py)
   |
Documents (documents/)
```
- **Web UI**: `templates/index.html`
- **API Endpoint**: `/rag` (POST)
- **Document Storage**: `documents/` folder

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Likithsatya192/Agentic-RAG-System
cd Agentic-RAG-System
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
# Or
source .venv/bin/activate  # On Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables
Create a `.env` file in the project root with:
```
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 5. Run the Application
```bash
python app.py
```
Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Usage

### Web UI
- Upload PDF, DOCX, or TXT files.
- Enter your question and submit.
- Results are shown below the form.
- If no documents are uploaded, the system will use web search.

### CLI (for development/testing)
You can invoke the backend logic directly by calling methods in `main.py` or using a custom CLI script.

---

## Environment Variables
- `GROQ_API_KEY`: API key for Groq LLM (required for LLM-based answers)
- `TAVILY_API_KEY`: API key for Tavily web search (required for web search)

---

## Advanced Customization
- **Add New Tools**: Extend `rag_tools.py` to add new retrieval or search tools.
- **Modify Agent Prompts**: Customize agent behavior in `agents.py`.
- **Change Workflow**: Edit `supervised_workflow.py` to adjust the research/analysis/writing pipeline.
- **Switch LLMs**: Swap out the LLM in `main.py` for another provider (OpenAI, Groq, etc.).
- **Production Deployment**: Use a WSGI server (e.g., Gunicorn) for deployment. Do not use Flask's dev server in production.

---

## Acknowledgments
- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Tavily](https://tavily.com/)
- [Groq](https://groq.com/)