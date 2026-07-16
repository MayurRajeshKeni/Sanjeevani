from langgraph.graph import StateGraph, END
from .state import RagAgentState
from .graph import should_continue

# Mock Nodes to trace routing paths without hitting live LLMs
def mock_retrieve(state: RagAgentState):
    print("[Mock Node: Retrieve]")
    return {"retrieved_chunks": [{"page_content": "mock content", "metadata": {}}]}

def mock_generate(state: RagAgentState):
    print("[Mock Node: Generate]")
    return {"draft_answer": "mock draft"}

def mock_critic(state: RagAgentState):
    # Simulate a failed groundedness check (score < 0.7)
    print("[Mock Node: Critic]")
    return {"critique_score": 0.2, "critique_feedback": "Hallucination detected."}

def mock_rewrite(state: RagAgentState):
    loop = state.get("loop_count", 0)
    print(f"[Mock Node: Rewrite] Healing Loop Retry {loop + 1}")
    return {"loop_count": loop + 1, "draft_answer": ""}

def mock_fallback(state: RagAgentState):
    print("[Mock Node: Fallback]")
    return {"draft_answer": "Information not found."}

def run_mock_test():
    print("\n--- Testing LangGraph Stateful Routing State Machine ---")
    
    # Assemble workflow graph with Mock nodes
    workflow = StateGraph(RagAgentState)
    
    workflow.add_node("retrieve", mock_retrieve)
    workflow.add_node("generate", mock_generate)
    workflow.add_node("critic", mock_critic)
    workflow.add_node("rewrite", mock_rewrite)
    workflow.add_node("fallback", mock_fallback)
    
    workflow.set_entry_point("retrieve")
    
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "critic")
    
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            END: END,
            "rewrite": "rewrite",
            "fallback": "fallback"
        }
    )
    
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("fallback", END)
    
    agent = workflow.compile()
    
    initial_state = {
        "original_query": "Test Query",
        "current_search_query": "Test Query",
        "retrieved_chunks": [],
        "draft_answer": "",
        "critique_score": 0.0,
        "critique_feedback": "",
        "loop_count": 0
    }
    
    final_state = agent.invoke(initial_state)
    print("\nFinal State Output:")
    print("Answer:", final_state["draft_answer"])
    print("Loop Count:", final_state["loop_count"])
    print("Groundedness Score:", final_state["critique_score"])
    
    # Assertions to ensure routing completed correctly
    assert final_state["draft_answer"] == "Information not found.", "Test failed: draft answer is not safety fallback."
    assert final_state["loop_count"] == 3, "Test failed: loop count did not reach 3 retries."
    print("\nSUCCESS: All routing state transitions verified successfully!")

if __name__ == "__main__":
    run_mock_test()
