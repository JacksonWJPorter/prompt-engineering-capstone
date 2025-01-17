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
    QualityCheckNode,
    HumanNode,
    FinalAnswerNode
)

# Initialize Memory to Persist State
checkpointer = MemorySaver()


def instantiate_nodes(initial_prompt: str):
    """Initialize all workflow nodes."""
    return {
        "human": HumanNode(initial_prompt),
        "original_prompt": OriginalPromptNode(initial_prompt),
        "categorize": CategorizePromptNode(initial_prompt),
        "disambiguation": QueryDisambiguationNode(initial_prompt),
        "rephrase": RephraseNode(initial_prompt),
        "enhancer": PromptEnhancerNode(""),
        "quality_check": QualityCheckNode(initial_prompt, ["clarity", "specificity"]),
        "final_answer_node": FinalAnswerNode("")
    }


def quality_router(state: GraphState) -> Literal["rephrase", "context", "end"]:
    """Router function to determine the next node based on quality check results."""
    quality_issues = state.keys.get("quality_issues", [])

    if "clarity" in quality_issues:
        return "rephrase"
    return "final_answer_node"


def disambiguous_router(state: GraphState):
    clarification = state.keys.get("clarification_question", "")
    if clarification == "clear":
        return "rephrase"
    return "human"


def build_workflow(nodes) -> StateGraph:
    """Build the workflow graph with all nodes and transitions."""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("human", nodes["human"])
    # workflow.add_node("original_prompt", nodes["original_prompt"])
    workflow.add_node("categorize", nodes["categorize"])
    workflow.add_node("disambiguation", nodes["disambiguation"])
    workflow.add_node("rephrase", nodes["rephrase"])
    workflow.add_node("enhancer", nodes["enhancer"])
    # workflow.add_node("quality_check", nodes["quality_check"])
    workflow.add_node("final_answer_node", nodes["final_answer_node"])

    # Define basic transitions
    # workflow.add_edge(START, "original_prompt")
    # workflow.add_edge("original_prompt", "categorize")
    workflow.add_edge(START, "categorize")
    workflow.add_edge("categorize", "disambiguation")

    # Add conditional edges after disambiguation
    workflow.add_conditional_edges(
        "disambiguation",
        disambiguous_router,
        {
            "rephrase": "rephrase",
            "human": "human"
        }
    )

    # Continue with rest of workflow
    workflow.add_edge("human", "disambiguation")
    workflow.add_edge("rephrase", "enhancer")
    # workflow.add_edge("enhancer", "quality_check")
    workflow.add_edge("enhancer", "final_answer_node")

    # # Add quality check routing
    # workflow.add_conditional_edges(
    #     "quality_check",
    #     quality_router,
    #     {
    #         "rephrase": "rephrase",
    #         "final_answer_node": "final_answer_node"
    #     }
    # )
    workflow.add_edge("final_answer_node", END)

    return workflow


class AgenticEnhancer:
    def __init__(self, initial_prompt: str):
        self.state = {"keys": {"original_prompt": initial_prompt}}

        # Initialize workflow
        nodes = instantiate_nodes(initial_prompt)
        self.workflow = build_workflow(nodes)
        self.app = self.workflow.compile(checkpointer=checkpointer)

    def format_final_state(self, final_state) -> dict:
        """Format the final state to match the desired structure."""

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

        original_prompt_answer = final_state["keys"].get(
            "original_prompt_answer", "No original answer provided")
        original_prompt_lin_probs = final_state["keys"].get(
            "original_prompt_lin_probs", "N/A")
        final_prompt_answer = final_state["keys"].get(
            "final_prompt_answer", "No final answer provided")
        final_prompt_lin_probs = final_state["keys"].get(
            "final_prompt_lin_probs", "N/A")
        enhanced_prompt = final_state["keys"].get("enhanced_prompt", "N/A")

        return {
            "original_prompt_answer": original_prompt_answer,
            "original_prompt_lin_probs": original_prompt_lin_probs,
            "final_prompt_answer": final_prompt_answer,
            "final_prompt_lin_probs": final_prompt_lin_probs,
            "enhanced_prompt": enhanced_prompt,
            # "enhancedConfig": enhancedConfig,
        }

    def execute_workflow(self):
        try:
            final_state = self.app.invoke(self.state, config={"configurable": {
                                          "thread_id": "jackson-test-chat-id"}})
            final_state_formatted = self.format_final_state(final_state)

            return final_state_formatted

        except Exception as e:
            print(colored(f"Error during workflow execution: {str(e)}",
                          'red', attrs=["bold"]))
            raise
