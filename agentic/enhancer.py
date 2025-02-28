# prompt_enhancer.py
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, Any, Literal
from termcolor import colored
from agentic.nodes import (
    GraphState,
    OriginalPromptNode,
    CategorizePromptNode,
    QueryDisambiguationNode,
    RephraseNode,
    PromptEnhancerNode,
    HumanNode,
    FinalAnswerNode,
    VersioningNode,
    PromptEvaluationNode
)

# Initialize Memory to Persist State
checkpointer = MemorySaver()


def instantiate_nodes(initial_prompt: str, event_emitter=None):
    """Initialize all workflow nodes."""
    return {
        "human": HumanNode(initial_prompt, event_emitter),
        "original_prompt": OriginalPromptNode(initial_prompt, event_emitter),
        "categorize": CategorizePromptNode(initial_prompt, event_emitter),
        "disambiguation": QueryDisambiguationNode(initial_prompt, event_emitter),
        "rephrase": RephraseNode(initial_prompt, event_emitter),
        "enhancer": PromptEnhancerNode("", event_emitter),
        "evaluator": PromptEvaluationNode(initial_prompt, event_emitter),
        "final_answer_node": FinalAnswerNode("", event_emitter),
        # Add versioning nodes for each step
        "v_categorize": VersioningNode("Categorizing Prompt", event_emitter),
        "v_disambiguation": VersioningNode("Checking for Ambiguity", event_emitter),
        "v_human": VersioningNode("Getting Human Feedback", event_emitter),
        "v_rephrase": VersioningNode("Rephrasing Prompt", event_emitter),
        "v_enhance": VersioningNode("Enhancing Prompt", event_emitter),
        "v_evaluate": VersioningNode("Evaluating Prompt", event_emitter),
        "v_final": VersioningNode("Generating Final Answer", event_emitter)
    }


def quality_router(state: GraphState) -> Literal["rephrase", "context", "end"]:
    """Router function to determine the next node based on quality check results."""
    quality_issues = state.keys.get("quality_issues", [])

    if "clarity" in quality_issues:
        return "rephrase"
    return "final_answer_node"


def disambiguous_router(state: GraphState):
    """Router function for disambiguation."""

    clarification = state.keys.get("clarification_question", "")
    if clarification == "clear":
        return "rephrase"
    return "human"


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
    # workflow.add_node("original_prompt", nodes["original_prompt"])
    workflow.add_node("human", nodes["human"])
    workflow.add_node("categorize", nodes["categorize"])
    workflow.add_node("disambiguation", nodes["disambiguation"])
    workflow.add_node("rephrase", nodes["rephrase"])
    workflow.add_node("enhancer", nodes["enhancer"])
    workflow.add_node("evaluator", nodes["evaluator"])
    workflow.add_node("final_answer_node", nodes["final_answer_node"])

    # Add versioning nodes
    workflow.add_node("v_categorize", nodes["v_categorize"])
    workflow.add_node("v_disambiguation", nodes["v_disambiguation"])
    workflow.add_node("v_human", nodes["v_human"])
    workflow.add_node("v_rephrase", nodes["v_rephrase"])
    workflow.add_node("v_enhance", nodes["v_enhance"])
    workflow.add_node("v_evaluate", nodes["v_evaluate"])
    workflow.add_node("v_final", nodes["v_final"])

    # Define transitions with versioning
    workflow.add_edge(START, "v_categorize")
    workflow.add_edge("v_categorize", "categorize")
    workflow.add_edge("categorize", "v_disambiguation")
    workflow.add_edge("v_disambiguation", "disambiguation")

    # Add conditional edges after disambiguation
    workflow.add_conditional_edges(
        "disambiguation",
        disambiguous_router,
        {
            "rephrase": "v_rephrase",
            "human": "v_human"
        }
    )

    # Continue with rest of workflow
    workflow.add_edge("v_human", "human")
    workflow.add_edge("human", "v_rephrase")
    workflow.add_edge("v_rephrase", "rephrase")
    workflow.add_edge("rephrase", "v_enhance")
    workflow.add_edge("v_enhance", "enhancer")
    workflow.add_edge("enhancer", "v_evaluate")
    workflow.add_edge("v_evaluate", "evaluator")

    # Add conditional routing based on evaluation
    workflow.add_conditional_edges(
        "evaluator",
        evaluation_router,
        {
            "rephrase": "v_rephrase",
            "final_answer_node": "v_final"
        }
    )

    workflow.add_edge("v_final", "final_answer_node")
    workflow.add_edge("final_answer_node", END)

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

        # TODO: Dynamic model config
        # final_ans_model_config = final_state["keys"].get(
        #     "final_ans_model_config", {})
        # enhancedConfig = {
        #     "maxOutputTokens": final_ans_model_config.get("MODEL_MAX_OUTPUT_TOKENS", 100),
        #     "name": final_ans_model_config.get("MODEL_NAME", "N/A"),
        #     "temperature": final_ans_model_config.get("MODEL_TEMPERATURE", "N/A"),
        #     "topK": final_ans_model_config.get("MODEL_TOP_K", "N/A"),
        #     "topP": final_ans_model_config.get("MODEL_TOP_P", "N/A")
        # }

    def format_final_state(self, final_state) -> dict:
        """Format the final state to match the desired structure."""
        original_prompt_answer = final_state["keys"].get(
            "original_prompt_answer", "No original answer provided")
        original_prompt_lin_probs = final_state["keys"].get(
            "original_prompt_lin_probs", "N/A")
        final_prompt_answer = final_state["keys"].get(
            "final_prompt_answer", "No final answer provided")
        final_prompt_lin_probs = final_state["keys"].get(
            "final_prompt_lin_probs", "N/A")
        enhanced_prompt = final_state["keys"].get("enhanced_prompt", "N/A")

        # Get evaluation results
        evaluation_results = final_state["keys"].get("prompt_evaluation", {})

        return {
            "original_prompt": self.initial_prompt,
            "original_prompt_answer": original_prompt_answer,
            "original_prompt_lin_probs": original_prompt_lin_probs,
            "final_prompt_answer": final_prompt_answer,
            "final_prompt_lin_probs": final_prompt_lin_probs,
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
