# prompt_enhancer.py
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, Any, Literal
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
    """Router function based on evaluation results."""
    evaluation = state.get_value("prompt_evaluation", {})
    needs_improvement = evaluation.get("needs_improvement", False)
    iteration_count = evaluation.get("iteration_count", 0)

    print(
        f"Evaluation router: needs_improvement={needs_improvement}, iteration_count={iteration_count}")

    # Either the prompt is good enough or we've tried enough times
    if not needs_improvement or iteration_count >= 3:
        return "final_answer_node"
    else:
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
            "needs_clarification": False,
            "category": None,
            "version": 1
        })

        # Initialize workflow
        nodes = instantiate_nodes(initial_prompt, event_emitter)
        self.workflow = build_workflow(nodes)
        self.app = self.workflow.compile(checkpointer=checkpointer)

    def format_final_state(self, final_state) -> dict:
        """Format the final state to match the desired structure."""
        enhanced_prompt = final_state["keys"].get("enhanced_prompt", "N/A")

        # Get evaluation results
        evaluation_results = final_state["keys"].get("prompt_evaluation", {})

        return {
            "original_prompt": self.initial_prompt,
            "enhanced_prompt": enhanced_prompt,
            "prompt_evaluation": {
                "overall_score": evaluation_results.get("overall_score"),
                "scores": evaluation_results.get("scores", {}),
                "feedback": evaluation_results.get("feedback", {}),
                "suggestions": evaluation_results.get("improvement_suggestions", []),
                "needs_improvement": evaluation_results.get("needs_improvement", False)
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
            final_state_formatted = self.format_final_state(final_state)

            return final_state_formatted

        except Exception as e:
            print(colored(f"Error during workflow execution: {str(e)}",
                          'red', attrs=["bold"]))
            raise
