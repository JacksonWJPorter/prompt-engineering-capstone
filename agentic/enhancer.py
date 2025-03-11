# prompt_enhancer.py
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, Any
from termcolor import colored
from agentic.nodes import (
    GraphState,
    CategorizePromptNode,
    RephraseNode,
    PromptEnhancerNode,
    PromptEvaluationNode
)

# Initialize Memory to Persist State
checkpointer = MemorySaver()


def instantiate_nodes(initial_prompt: str, event_emitter=None):
    """Initialize all workflow nodes."""
    return {
        "categorize": CategorizePromptNode(initial_prompt, event_emitter),
        "rephrase": RephraseNode(initial_prompt, event_emitter),
        "enhancer": PromptEnhancerNode("", event_emitter),
        "evaluator": PromptEvaluationNode(initial_prompt, event_emitter),
    }


def evaluation_router(state: GraphState):
    """Route based on evaluation score."""
    evaluation = state.get_value("prompt_evaluation", {})
    needs_improvement = evaluation.get("needs_improvement", False)
    iteration_count = evaluation.get("iteration_count", 0)
    score = evaluation.get("score", 0)

    # Either the prompt is good enough (>= 75) or we've tried 3 times
    if score >= 75 or iteration_count >= 3:
        return "final_answer_node"
    return "rephrase"


def build_workflow(nodes) -> StateGraph:
    """Build the workflow graph with all nodes and transitions."""
    workflow = StateGraph(GraphState)

    # Add all nodes to the workflow
    workflow.add_node("categorize", nodes["categorize"])
    workflow.add_node("rephrase", nodes["rephrase"])
    workflow.add_node("enhancer", nodes["enhancer"])
    workflow.add_node("evaluator", nodes["evaluator"])

    # Define transitions with versioning
    workflow.add_edge(START, "categorize")
    workflow.add_edge("categorize", "rephrase")
    workflow.add_edge("rephrase", "enhancer")
    workflow.add_edge("enhancer", "evaluator")
    workflow.add_edge("evaluator", END)

    return workflow


class AgenticEnhancer:
    def __init__(self, initial_prompt: str, event_emitter=None):
        self.initial_prompt = initial_prompt
        self.event_emitter = event_emitter
        self.state = GraphState.create_initial_state({
            "original_prompt": initial_prompt,
            "enhanced_prompt": initial_prompt,
            "category": None,
            "iteration_count": 0
        })

        # Initialize workflow
        nodes = instantiate_nodes(initial_prompt, event_emitter)
        self.workflow = build_workflow(nodes)
        self.app = self.workflow.compile(checkpointer=checkpointer)

    def format_final_state(self, final_state) -> dict:
        """Format the final state with evaluation results."""
        enhanced_prompt = final_state["keys"].get("enhanced_prompt", "N/A")
        evaluation = final_state["keys"].get("prompt_evaluation", {})
        
        return {
            "original_prompt": self.initial_prompt,
            "enhanced_prompt": enhanced_prompt,
            "evaluation": {
                "score": evaluation.get("score"),
                "justification": evaluation.get("justification"),
                "suggestions": evaluation.get("suggestions", []),
                "iteration_count": evaluation.get("iteration_count", 0)
            }
        }

    def execute_workflow(self):
        try:
            # Add configuration for checkpointer
            config = {
                "configurable": {
                    "thread_id": "agentic-workflow",  # Unique thread ID
                    "checkpoint_ns": "prompt-enhancement",  # Namespace for checkpoints
                    # Unique ID for this run
                    "checkpoint_id": f"prompt-{hash(self.initial_prompt)}"
                }
            }

            final_state = self.app.invoke(self.state, config=config)
            return self.format_final_state(final_state)

        except Exception as e:
            print(colored(f"Error during workflow execution: {str(e)}",
                          'red', attrs=["bold"]))
            raise
