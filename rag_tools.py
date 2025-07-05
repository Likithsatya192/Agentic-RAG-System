from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
from langchain_community.vectorstores import FAISS
import os
import sys

try:
    from langchain_tavily import TavilySearch
    TAVILY_WRAPPER_CLASS = TavilySearch
    print("Using TavilySearch from langchain_tavily.")
except ImportError as e:
    print("[WARNING] Could not import TavilySearch from langchain_tavily. Falling back to deprecated version.", file=sys.stderr)
    print(f"[WARNING] ImportError: {e}", file=sys.stderr)
    try:
        from langchain_community.tools import TavilySearchResults
        TAVILY_WRAPPER_CLASS = TavilySearchResults
    except ImportError:
        TAVILY_WRAPPER_CLASS = None

from pydantic import BaseModel, Field

class RAGRetrievalTool(BaseTool):
    """Tool for retrieving information from local RAG system"""
    
    name: str = "rag_retrieval"
    description: str = "Retrieve relevant information from local documents using RAG. Use this tool to search through local knowledge base for information related to the query."
    vector_store: FAISS = Field(exclude=True)
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, vector_store: FAISS, **kwargs):
        super().__init__(vector_store=vector_store, **kwargs)
    
    def _run(self, query: str, k: int = 5) -> str:
        """Retrieve relevant documents for the query"""
        try:
            if not self.vector_store:
                return "Vector store not available"
            
            # Perform similarity search
            docs = self.vector_store.similarity_search(query, k=k)
            
            if not docs:
                return "No relevant documents found in local knowledge base"
            
            # Format results
            results = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                results.append(f"Document {i} (Source: {source}):\n{content}")
            
            return f"Found {len(docs)} relevant documents:\n\n" + "\n\n".join(results)
            
        except Exception as e:
            return f"Error retrieving documents: {str(e)}"
    
    async def _arun(self, query: str, k: int = 5) -> str:
        """Async version of _run"""
        return self._run(query, k)

class WebSearchTool(BaseTool):
    """Tool for web search when local RAG is insufficient"""
    
    name: str = "web_search"
    description: str = "Search the web for information when local documents are insufficient. Use this tool to find current information from the internet."
    tavily_api_key: str = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    
    def __init__(self, tavily_api_key: str = None, **kwargs):
        if tavily_api_key is None:
            tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        super().__init__(tavily_api_key=tavily_api_key, **kwargs)
    
    def _run(self, query: str) -> str:
        """Perform web search"""
        try:
            if not self.tavily_api_key:
                return "Web search not available - TAVILY_API_KEY not provided"
            
            if TAVILY_WRAPPER_CLASS is None:
                return "Web search not available - Tavily package not installed"
            
            # Initialize Tavily search
            search = TAVILY_WRAPPER_CLASS(api_key=self.tavily_api_key)
            
            # Perform search
            results = search.run(query)
            
            if not results:
                return "No web search results found"
            
            return f"Web search results for '{query}':\n\n{results}"
            
        except Exception as e:
            return f"Error performing web search: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version of _run"""
        return self._run(query)

class MockRAGTool(BaseTool):
    """Mock RAG tool for testing when no vector store is available"""
    
    name: str = "rag_retrieval"
    description: str = "Retrieve relevant information from local documents using RAG"
    
    def _run(self, query: str) -> str:
        """Mock retrieval for testing"""
        return f"Mock RAG response for query: {query}. This is a placeholder response since no actual vector store is configured."
    
    async def _arun(self, query: str) -> str:
        """Async version of _run"""
        return self._run(query)

class MockWebSearchTool(BaseTool):
    """Mock web search tool for testing when no API key is available"""
    
    name: str = "web_search"
    description: str = "Search the web for information when local documents are insufficient"
    
    def _run(self, query: str) -> str:
        """Mock web search for testing"""
        return f"Mock web search response for query: {query}. This is a placeholder response since no actual web search API is configured."
    
    async def _arun(self, query: str) -> str:
        """Async version of _run"""
        return self._run(query)

def create_tools(vector_store: FAISS = None, tavily_api_key: str = None) -> List[BaseTool]:
    """Create and return a list of tools"""
    tools = []
    
    # Add RAG tool
    if vector_store is not None:
        tools.append(RAGRetrievalTool(vector_store=vector_store))
    else:
        tools.append(MockRAGTool())
    
    # Add web search tool
    if tavily_api_key or os.getenv("TAVILY_API_KEY"):
        tools.append(WebSearchTool(tavily_api_key=tavily_api_key))
    else:
        tools.append(MockWebSearchTool())
    
    return tools