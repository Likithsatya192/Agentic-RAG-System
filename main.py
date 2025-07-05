import os
from typing import Dict, Any
from langchain_groq import ChatGroq
from document_loader import DocumentLoader
from rag_tools import RAGRetrievalTool, WebSearchTool
from agents import ResearchAgent, AnalysisAgent, WriterAgent
from supervised_workflow import SupervisedRAGWorkflow
from dotenv import load_dotenv
load_dotenv()

class AgenticRAGSystem:
    """Main system class that orchestrates everything"""
    
    def __init__(self, 
                 documents_directory: str,
                 groq_api_key: str,
                 tavily_api_key: str = None,
                 model_name: str = "llama3-8b-8192"):
        
        self.documents_directory = documents_directory
        self.groq_api_key = groq_api_key
        self.tavily_api_key = tavily_api_key
        self.model_name = model_name
        
        # Initialize components
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name=model_name,
            temperature=0.1
        )
        
        self.document_loader = DocumentLoader()
        self.vector_store = None
        self.workflow = None
        
        # Initialize the system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the RAG system and agents"""
        print("Initializing Agentic RAG System...")
        
        # Load documents and create vector store
        print(f"Loading documents from {self.documents_directory}")
        documents = self.document_loader.load_documents_from_directory(
            self.documents_directory
        )
        
        if documents:
            self.vector_store = self.document_loader.create_vector_store(documents)
            print(f"Vector store created with {len(documents)} documents")
        else:
            print("No documents found or loaded")
        
        # Initialize tools
        tools = []
        if self.vector_store:
            tools.append(RAGRetrievalTool(vector_store=self.vector_store))
        
        if self.tavily_api_key:
            tools.append(WebSearchTool(tavily_api_key=self.tavily_api_key))
        
        # Initialize agents
        research_agent = ResearchAgent(self.llm, tools)
        analysis_agent = AnalysisAgent(self.llm)
        writer_agent = WriterAgent(self.llm)
        
        # Initialize workflow
        self.workflow = SupervisedRAGWorkflow(
            research_agent=research_agent,
            analysis_agent=analysis_agent,
            writer_agent=writer_agent
        )
        
        print("System initialized successfully!")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process a query through the agentic RAG system"""
        if not self.workflow:
            return {"error": "System not properly initialized"}
        
        try:
            result = self.workflow.run(question)
            return result
        except Exception as e:
            return {"error": f"Query processing failed: {str(e)}"}
    
    def add_documents(self, new_documents_directory: str):
        """Add new documents to the system"""
        print(f"Adding documents from {new_documents_directory}")
        
        new_documents = self.document_loader.load_documents_from_directory(
            new_documents_directory
        )
        
        if new_documents:
            if self.vector_store:
                # Add to existing vector store
                new_vector_store = self.document_loader.create_vector_store(new_documents)
                self.vector_store.merge_from(new_vector_store)
            else:
                # Create new vector store
                self.vector_store = self.document_loader.create_vector_store(new_documents)
            
            print(f"Added {len(new_documents)} new documents")
        else:
            print("No new documents found")

if __name__ == "__main__":
    # Configuration
    DOCUMENTS_DIR = "documents"  # Directory containing PDF, DOC, TXT files
    
    # Initialize system
    rag_system = AgenticRAGSystem(
        documents_directory=DOCUMENTS_DIR,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )
    
    # Example queries
    queries = [
        "What are the main findings in the research documents?",
        "Summarize the key points about machine learning from the available documents",
        "What recent developments in AI are mentioned in the documents?"
    ]
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print('='*50)
        
        result = rag_system.query(query)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Final Output: {result['final_output']}")
            print(f"Iterations: {result['iterations']}")
            print(f"Used Web Search: {result['used_web_search']}")