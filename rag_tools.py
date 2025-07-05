from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
from langchain_community.vectorstores import FAISS
try:
    from langchain_tavily import TavilySearchResults  # New recommended import
except ImportError:
    from langchain_community.tools import TavilySearchResults  # Fallback for older versions
from pydantic import BaseModel, Field
import os

class RAGRetrievalTool(BaseTool):
    """Tool for retrieving information from local RAG system"""
    
    name: str = "rag_retrieval"
    description: str = "Retrieve relevant information from local documents using RAG"
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
            
            return "\n\n".join(results)
            
        except Exception as e:
            return f"Error retrieving documents: {str(e)}"
    
    async def _arun(self, query: str, k: int = 5) -> str:
        """Async version of _run"""
        return self._run(query, k)

class WebSearchTool(BaseTool):
    """Tool for web search when local RAG is insufficient"""
    
    name: str = "web_search"
    description: str = "Search the web for information when local documents are insufficient"
    
    def __init__(self, tavily_api_key: str = os.getenv("TAVILY_API_KEY"), **kwargs):
        super().__init__(**kwargs)
        # Avoid Pydantic field error by using object.__setattr__
        object.__setattr__(self, "search_tool", TavilySearchResults(
            api_key=tavily_api_key,
            max_results=5
        ) if tavily_api_key else None)
    
    def _run(self, query: str) -> str:
        """Perform web search"""
        try:
            if not getattr(self, "search_tool", None):
                return "Web search not available - API key not provided"
            
            results = self.search_tool.run(query)
            
            if not results:
                return "No web search results found"
            
            formatted_results = []
            for result in results:
                if isinstance(result, dict):
                    title = result.get('title', 'No title')
                    content = result.get('content', 'No content')
                    url = result.get('url', 'No URL')
                    formatted_results.append(f"Title: {title}\nContent: {content}\nURL: {url}")
                else:
                    formatted_results.append(str(result))
            
            return "\n\n".join(formatted_results)
            
        except Exception as e:
            return f"Error performing web search: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version of _run"""
        return self._run(query)