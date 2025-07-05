from typing import List, Dict, Any, Optional
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.tools import BaseTool

class ResearchAgent:
    """Agent responsible for research using RAG and web search"""
    
    def __init__(self, llm: ChatGroq, tools: List[BaseTool]):
        self.llm = llm
        self.tools = tools
        
        # System prompt for research
        system_prompt = """You are a Research Agent specialized in finding relevant information.
        
        Your task is to:
        1. ALWAYS start by using the 'rag_retrieval' tool to search local documents first
        2. If local information is insufficient or empty, then use the 'web_search' tool
        3. Provide comprehensive research results from your searches
        4. Always indicate the source of information (local documents vs web search)
        5. Be thorough but concise in your research
        
        IMPORTANT: You must actually USE the tools provided to you. Don't just describe what you would do - actually call the tools to get information.
        
        When you receive a query, follow this process:
        1. First, call the rag_retrieval tool with the query
        2. Analyze the results from local documents
        3. If the local results are insufficient, call the web_search tool
        4. Provide a comprehensive summary of all findings
        """
        
        # Create the react agent
        self.agent = create_react_agent(
            self.llm,
            self.tools
        )
    
    def research(self, query: str, prefer_web: bool = False) -> Dict[str, Any]:
        """Perform research based on query"""
        try:
            # Create the input with clear instructions
            if prefer_web:
                research_instruction = f"""Please research the following query, prioritizing web search: {query}
                
                1. First try the rag_retrieval tool to check local documents
                2. Then use web_search tool for current information
                3. Provide comprehensive results from both sources if available"""
            else:
                research_instruction = f"""Please research the following query: {query}
                
                1. Start with the rag_retrieval tool to search local documents
                2. If local information is insufficient, use the web_search tool
                3. Provide comprehensive results indicating sources used"""
            
            # Invoke the agent
            result = self.agent.invoke({
                "messages": [HumanMessage(content=research_instruction)]
            })
            
            # Extract the final response
            messages = result.get("messages", [])
            output = ""
            
            for msg in reversed(messages):
                if hasattr(msg, 'content') and hasattr(msg, 'type'):
                    if msg.type == "ai":
                        output = msg.content
                        break
                elif isinstance(msg, dict):
                    if msg.get("type") == "ai" or msg.get("role") == "assistant":
                        output = msg.get("content", "")
                        break
            
            # Determine which sources were used based on the output
            sources_used = []
            if "local documents" in output.lower() or "rag" in output.lower():
                sources_used.append("local")
            if "web search" in output.lower() or "internet" in output.lower():
                sources_used.append("web")
            
            return {
                "query": query,
                "result": output,
                "sources_used": sources_used,
                "success": True,
                "raw_messages": messages  # For debugging
            }
            
        except Exception as e:
            return {
                "query": query,
                "result": f"Research failed: {str(e)}",
                "sources_used": [],
                "success": False,
                "error": str(e)
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
            SUFFICIENT: Yes/No
            QUALITY: High/Medium/Low
            GAPS: List any information gaps (or "None" if no gaps)
            RECOMMENDATION: Next steps (proceed_to_writing/need_web_search/need_more_research)
            REASONING: Brief explanation of your decision
            
            Be critical but fair in your assessment."""),
            ("user", "User Query: {query}\n\nResearch Results:\n{research_results}")
        ])
    
    def analyze(self, query: str, research_results: str) -> Dict[str, Any]:
        """Analyze research results and determine next steps"""
        try:
            # Format and invoke the prompt
            messages = self.prompt.format_messages(
                query=query,
                research_results=research_results
            )
            
            response = self.llm.invoke(messages)
            analysis_text = response.content
            
            # Parse the structured response
            analysis = {}
            lines = analysis_text.split('\n')
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    analysis[key.strip()] = value.strip()
            
            # Determine if sufficient
            sufficient = analysis.get('SUFFICIENT', 'No').lower() == 'yes'
            
            return {
                "sufficient": sufficient,
                "quality": analysis.get('QUALITY', 'Unknown'),
                "gaps": analysis.get('GAPS', ''),
                "recommendation": analysis.get('RECOMMENDATION', 'need_more_research'),
                "reasoning": analysis.get('REASONING', ''),
                "full_analysis": analysis_text,
                "success": True
            }
            
        except Exception as e:
            return {
                "sufficient": False,
                "quality": "Unknown",
                "gaps": "Analysis failed",
                "recommendation": "need_more_research",
                "reasoning": f"Error during analysis: {str(e)}",
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
            - Clear introduction that addresses the user's query
            - Key findings organized logically
            - Proper source citations (distinguish between local documents and web sources)
            - Conclusion that directly answers the user's question
            
            Make your response informative and easy to read."""),
            ("user", "User Query: {query}\n\nResearch Results:\n{research_results}\n\nAnalysis Summary:\n{analysis}")
        ])
    
    def write(self, query: str, research_results: str, analysis: str) -> Dict[str, Any]:
        """Write final summary based on research and analysis"""
        try:
            # Format and invoke the prompt
            messages = self.prompt.format_messages(
                query=query,
                research_results=research_results,
                analysis=analysis
            )
            
            response = self.llm.invoke(messages)
            
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

class MultiAgentOrchestrator:
    """Orchestrates the multi-agent workflow"""
    
    def __init__(self, llm: ChatGroq, tools: List[BaseTool]):
        self.research_agent = ResearchAgent(llm, tools)
        self.analysis_agent = AnalysisAgent(llm)
        self.writer_agent = WriterAgent(llm)
        self.tools = tools
    
    def process_query(self, query: str, max_iterations: int = 3) -> Dict[str, Any]:
        """Process a query through the multi-agent system"""
        
        workflow_log = []
        iteration = 0
        
        # Start with local search preference
        prefer_web = False
        
        while iteration < max_iterations:
            iteration += 1
            workflow_log.append(f"=== Iteration {iteration} ===")
            
            # Research phase
            workflow_log.append("Starting research phase...")
            research_result = self.research_agent.research(query, prefer_web=prefer_web)
            
            if not research_result["success"]:
                workflow_log.append(f"Research failed: {research_result.get('error', 'Unknown error')}")
                break
            
            workflow_log.append(f"Research completed. Sources used: {research_result.get('sources_used', [])}")
            
            # Analysis phase
            workflow_log.append("Starting analysis phase...")
            analysis_result = self.analysis_agent.analyze(query, research_result["result"])
            
            if not analysis_result["success"]:
                workflow_log.append("Analysis failed")
                break
            
            workflow_log.append(f"Analysis completed. Sufficient: {analysis_result['sufficient']}")
            workflow_log.append(f"Recommendation: {analysis_result['recommendation']}")
            
            # Check if we have sufficient information
            if analysis_result["sufficient"]:
                workflow_log.append("Information deemed sufficient. Proceeding to writing phase...")
                
                # Writing phase
                write_result = self.writer_agent.write(
                    query,
                    research_result["result"],
                    analysis_result["full_analysis"]
                )
                
                return {
                    "query": query,
                    "final_answer": write_result["summary"],
                    "research_results": [research_result],
                    "analysis": analysis_result,
                    "iterations": iteration,
                    "success": write_result["success"],
                    "workflow_log": workflow_log
                }
            
            # Check recommendation for next steps
            recommendation = analysis_result["recommendation"].lower()
            if "web_search" in recommendation and not prefer_web:
                workflow_log.append("Switching to web search preference for next iteration...")
                prefer_web = True
                continue
            elif "more_research" in recommendation:
                workflow_log.append("Need more research. Continuing...")
                continue
            else:
                workflow_log.append("No clear next steps. Breaking...")
                break
        
        # If we exit the loop without sufficient information
        workflow_log.append("Maximum iterations reached or no more research options available")
        workflow_log.append("Proceeding to write with available information...")
        
        # Final attempt at writing with whatever we have
        final_research = research_result["result"] if 'research_result' in locals() else "No research results available"
        final_analysis = analysis_result.get("full_analysis", "No analysis available") if 'analysis_result' in locals() else "No analysis available"
        
        write_result = self.writer_agent.write(query, final_research, final_analysis)
        
        return {
            "query": query,
            "final_answer": write_result["summary"],
            "research_results": [research_result] if 'research_result' in locals() else [],
            "analysis": analysis_result if 'analysis_result' in locals() else {},
            "iterations": iteration,
            "success": False,
            "note": "Maximum iterations reached or insufficient information found",
            "workflow_log": workflow_log
        }

def create_multi_agent_system(llm: ChatGroq, tools: List[BaseTool]) -> MultiAgentOrchestrator:
    """Create and return a multi-agent system"""
    return MultiAgentOrchestrator(llm, tools)

# Example usage and testing function
def test_system(llm: ChatGroq, tools: List[BaseTool], test_query: str = "What is machine learning?"):
    """Test the multi-agent system"""
    orchestrator = create_multi_agent_system(llm, tools)
    
    print(f"Testing query: {test_query}")
    print("=" * 50)
    
    result = orchestrator.process_query(test_query)
    
    print("WORKFLOW LOG:")
    for log_entry in result.get("workflow_log", []):
        print(f"  {log_entry}")
    
    print("\nFINAL RESULT:")
    print(f"Success: {result['success']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Answer: {result['final_answer']}")
    
    return result