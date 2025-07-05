from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
import logging
from agents import ResearchAgent, AnalysisAgent, WriterAgent
import uuid
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkflowState(BaseModel):
    """State object for the workflow"""
    query: str
    research_results: str = ""
    analysis_results: Dict[str, Any] = {}
    final_output: str = ""
    iteration_count: int = 0
    max_iterations: int = 3
    use_web_search: bool = False
    satisfied: bool = False
    
    class Config:
        arbitrary_types_allowed = True

class SupervisedRAGWorkflow:
    """Supervised multi-agent RAG workflow using LangGraph"""
    
    def __init__(self, research_agent: ResearchAgent, analysis_agent: AnalysisAgent, 
                 writer_agent: WriterAgent):
        self.research_agent = research_agent
        self.analysis_agent = analysis_agent
        self.writer_agent = writer_agent
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
        self.memory = MemorySaver()
        
        # Compile the graph
        self.app = self.workflow.compile(checkpointer=self.memory)
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("research", self._research_node)
        workflow.add_node("analysis", self._analysis_node)
        workflow.add_node("writer", self._writer_node)
        workflow.add_node("supervisor", self._supervisor_node)
        
        # Add edges
        workflow.set_entry_point("research")
        workflow.add_edge("research", "analysis")
        workflow.add_edge("analysis", "supervisor")
        workflow.add_conditional_edges(
            "supervisor",
            self._should_continue,
            {
                "continue": "research",
                "write": "writer",
                "end": END
            }
        )
        workflow.add_edge("writer", END)
        
        return workflow
    
    def _research_node(self, state: WorkflowState) -> WorkflowState:
        """Research node - performs research using appropriate method"""
        logger.info(f"Research node - Iteration {state.iteration_count + 1}")
        
        research_result = self.research_agent.research(
            state.query, 
            prefer_web=state.use_web_search
        )
        
        state.research_results = research_result["result"]
        state.iteration_count += 1
        
        logger.info(f"Research completed. Success: {research_result['success']}")
        return state
    
    def _analysis_node(self, state: WorkflowState) -> WorkflowState:
        """Analysis node - analyzes research results"""
        logger.info("Analysis node - Evaluating research results")
        
        analysis_result = self.analysis_agent.analyze(
            state.query,
            state.research_results
        )
        
        state.analysis_results = analysis_result
        
        logger.info(f"Analysis completed. Sufficient: {analysis_result.get('sufficient', False)}")
        return state
    
    def _writer_node(self, state: WorkflowState) -> WorkflowState:
        """Writer node - creates final output"""
        logger.info("Writer node - Creating final output")
        
        writing_result = self.writer_agent.write(
            state.query,
            state.research_results,
            state.analysis_results.get("full_analysis", "")
        )
        
        state.final_output = writing_result["summary"]
        
        logger.info(f"Writing completed. Success: {writing_result['success']}")
        return state
    
    def _supervisor_node(self, state: WorkflowState) -> WorkflowState:
        """Supervisor node - makes decisions about next steps"""
        logger.info("Supervisor node - Making decision")
        
        analysis = state.analysis_results
        
        # Check if we should continue, write, or end
        if analysis.get("sufficient", False):
            logger.info("Supervisor decision: Proceed to writing")
            return state
        elif state.iteration_count >= state.max_iterations:
            logger.info("Supervisor decision: Max iterations reached, proceeding to writing")
            return state
        elif not state.use_web_search and "web search" in analysis.get("recommendation", "").lower():
            logger.info("Supervisor decision: Escalating to web search")
            state.use_web_search = True
            return state
        else:
            logger.info("Supervisor decision: Continue research")
            return state
    
    def _should_continue(self, state: WorkflowState) -> str:
        """Determine next action based on state"""
        analysis = state.analysis_results
        
        if analysis.get("sufficient", False):
            return "write"
        elif state.iteration_count >= state.max_iterations:
            return "write"
        else:
            return "continue"
    
    def run(self, query: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the complete workflow"""
        logger.info(f"Starting workflow for query: {query}")
        
        # Initialize state
        initial_state = WorkflowState(query=query)
        
        # Ensure config has a unique thread_id for the checkpointer
        if config is None:
            config = {}
        if "configurable" not in config:
            config["configurable"] = {}
        if not any(k in config["configurable"] for k in ("thread_id", "checkpoint_ns", "checkpoint_id")):
            config["configurable"]["thread_id"] = str(uuid.uuid4())
        # Run the workflow
        final_state = self.app.invoke(initial_state, config=config)
        
        logger.info("Workflow completed")
        
        return {
            "query": query,
            "final_output": final_state.get("final_output", ""),
            "research_results": final_state.get("research_results", ""),
            "analysis_results": final_state.get("analysis_results", {}),
            "iterations": final_state.get("iteration_count", 0),
            "used_web_search": final_state.get("use_web_search", False)
        }