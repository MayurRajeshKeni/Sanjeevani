from langgraph.graph import StateGraph, END
from .state import RagAgentState
from .nodes import AgentNodes
from src.retrieval.fusion import HybridRetriever

def should_continue(state: RagAgentState) -> str:
    """Conditional routing logic executed after the Critic Node evaluation.
    
    Args:
        state: The current LangGraph agent state.
        
    Returns:
        The target node name to route to next.
    """
    score = state.get("critique_score", 0.0)
    loop_count = state.get("loop_count", 0)
    
    if score >= 0.7:
        print(f"\n[Routing] Groundedness score {score:.2f} >= 0.7. ROUTING TO END.")
        return END
        
    if loop_count >= 3:
        print(f"\n[Routing] Groundedness score {score:.2f} < 0.7 and loop_count {loop_count} >= 3. ROUTING TO FALLBACK.")
        return "fallback"
        
    print(f"\n[Routing] Groundedness score {score:.2f} < 0.7 and loop_count {loop_count} < 3. ROUTING TO REWRITE.")
    return "rewrite"

def create_agent_graph(retriever: HybridRetriever):
    """Builds and compiles the self-healing state machine workflow using LangGraph.
    
    Args:
        retriever: Injectable HybridRetriever instance.
        
    Returns:
        Compiled LangGraph runner.
    """
    # Instantiate the node functions with dependency injection
    nodes = AgentNodes(retriever)
    
    # Initialize the state-driven workflow graph
    workflow = StateGraph(RagAgentState)
    
    # 1. Register workflow nodes
    workflow.add_node("retrieve", nodes.retrieve_node)
    workflow.add_node("generate", nodes.generate_node)
    workflow.add_node("critic", nodes.critic_node)
    workflow.add_node("rewrite", nodes.rewrite_node)
    workflow.add_node("fallback", nodes.fallback_node)
    
    # 2. Configure Entry Point
    workflow.set_entry_point("retrieve")
    
    # 3. Add standard directed edges
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "critic")
    
    # 4. Add conditional routing edges originating from the Critic Node
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            END: END,
            "rewrite": "rewrite",
            "fallback": "fallback"
        }
    )
    
    # 5. Complete cyclic loops and safety endings
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("fallback", END)
    
    # 6. Compile and return graph executor
    return workflow.compile()
