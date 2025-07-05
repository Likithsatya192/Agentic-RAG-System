from typing import List, Dict, Any, Optional
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.tools import BaseTool

class ResearchAgent:
    """Agent responsible for research using RAG and web search"""
    
    def __init__(self, llm: ChatGroq, tools: List[BaseTool]):
        self.llm = llm
        self.tools = tools
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Research Agent specialized in finding relevant information.
            Your task is to:
            1. First search local documents using RAG retrieval
            2. If local information is insufficient, use web search
            3. Provide comprehensive research results
            4. Always indicate the source of information
            
            Be thorough but concise in your research."""),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent
        self.agent = create_react_agent(self.llm, self.tools, prompt=self.prompt)
    
    def research(self, query: str, use_web: bool = False) -> Dict[str, Any]:
        """Perform research based on query"""
        try:
            # Modify query based on search preference
            if use_web:
                research_query = f"Search web for: {query}"
            else:
                research_query = f"Search local documents for: {query}"
            
            result = self.agent.invoke({"messages": [{"role": "user", "content": research_query}]})
            
            # Extract the last assistant message as the result
            messages = result.get("messages", [])
            output = ""
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    output = msg.get("content", "")
                    break
            return {
                "query": query,
                "result": output,
                "source": "web" if use_web else "local",
                "success": True
            }
            
        except Exception as e:
            return {
                "query": query,
                "result": f"Research failed: {str(e)}",
                "source": "web" if use_web else "local",
                "success": False
            }

class AnalysisAgent:
    """Agent responsible for analyzing research results and determining next steps"""
    
    def __init__(self, llm: ChatGroq):
        self.llm = llm
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an Analysis Agent specialized in evaluating research results.
            Your task is to:
            1. Analyze the quality and completeness of research results
            2. Determine if the information is sufficient to answer the user's query
            3. Decide if additional research is needed
            4. Provide recommendations for next steps
            
            Return your analysis in the following format:
            - SUFFICIENT: Yes/No
            - QUALITY: High/Medium/Low
            - GAPS: List any information gaps
            - RECOMMENDATION: Next steps (continue to writing, need more research, escalate to web search)
            - REASONING: Brief explanation of your decision"""),
            ("user", "User Query: {query}\n\nResearch Results: {research_results}"),
        ])
    
    def analyze(self, query: str, research_results: str) -> Dict[str, Any]:
        """Analyze research results and determine next steps"""
        try:
            response = self.llm.invoke(
                self.prompt.format_messages(
                    query=query,
                    research_results=research_results
                )
            )
            
            analysis_text = response.content
            
            # Parse the analysis (simple parsing - could be improved)
            lines = analysis_text.split('\n')
            analysis = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    analysis[key.strip().replace('- ', '')] = value.strip()
            
            # Determine if sufficient
            sufficient = analysis.get('SUFFICIENT', 'No').lower() == 'yes'
            
            return {
                "sufficient": sufficient,
                "quality": analysis.get('QUALITY', 'Unknown'),
                "gaps": analysis.get('GAPS', ''),
                "recommendation": analysis.get('RECOMMENDATION', ''),
                "reasoning": analysis.get('REASONING', ''),
                "full_analysis": analysis_text,
                "success": True
            }
            
        except Exception as e:
            return {
                "sufficient": False,
                "quality": "Unknown",
                "gaps": "",
                "recommendation": "Error in analysis",
                "reasoning": str(e),
                "full_analysis": "",
                "success": False
            }

class WriterAgent:
    """Agent responsible for writing final summaries and outputs"""
    
    def __init__(self, llm: ChatGroq):
        self.llm = llm
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Writer Agent specialized in creating clear, comprehensive summaries.
            Your task is to:
            1. Synthesize research results into a coherent response
            2. Extract and present key points clearly
            3. Maintain source attribution
            4. Ensure the output directly addresses the user's query
            
            Create a well-structured response with:
            - Clear introduction
            - Key findings organized logically
            - Proper source citations
            - Conclusion that directly answers the user's question"""),
            ("user", "User Query: {query}\n\nResearch Results: {research_results}\n\nAnalysis: {analysis}"),
        ])
    
    def write(self, query: str, research_results: str, analysis: str) -> Dict[str, Any]:
        """Write final summary based on research and analysis"""
        try:
            response = self.llm.invoke(
                self.prompt.format_messages(
                    query=query,
                    research_results=research_results,
                    analysis=analysis
                )
            )
            
            return {
                "query": query,
                "summary": response.content,
                "success": True
            }
            
        except Exception as e:
            return {
                "query": query,
                "summary": f"Writing failed: {str(e)}",
                "success": False
            }