# prompt_enhancer.py
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, Any
from termcolor import colored
from agentic.nodes import (
    HumanNode,
    QueryDisambiguationNode,
    GraphState,
    CategorizePromptNode,
    RephraseNode,
    PromptEnhancerNode,
    PromptEvaluationNode
)

# Initialize Memory to Persist State
checkpointer = MemorySaver()


def disambiguous_router(state: GraphState):
    """Router function for disambiguation."""

    clarification = state.keys.get("clarification_question", "")
    if clarification == "clear":
        return "rephrase"
    return "human"


def instantiate_nodes(initial_prompt: str, event_emitter=None):
    """Initialize all workflow nodes."""
    return {
        "categorize": CategorizePromptNode(initial_prompt, event_emitter),
        "rephrase": RephraseNode(initial_prompt, event_emitter),
        "enhancer": PromptEnhancerNode("", event_emitter),
        "evaluator": PromptEvaluationNode(initial_prompt, event_emitter),
        "human": HumanNode(initial_prompt, event_emitter),
        "disambiguation": QueryDisambiguationNode(initial_prompt, event_emitter),
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
    workflow.add_node("human", nodes["human"])
    workflow.add_node("disambiguation", nodes["disambiguation"])

    # Define transitions with versioning
    workflow.add_edge(START, "disambiguation")
    workflow.add_edge(START, "categorize")
    workflow.add_conditional_edges(
        "disambiguation",
        disambiguous_router,
        {
            "rephrase": "rephrase",
            "human": "human"
        }
    )
    workflow.add_edge("human", "disambiguation")
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

    def clean_markdown(self, text):
        """Remove all markdown formatting from text."""
        if not text or not isinstance(text, str):
            return text

        # Remove common markdown formatting
        text = text.replace('**', '')
        text = text.replace('*', '')

        # Remove headings (# to ######)
        for i in range(1, 7):
            heading_marker = '#' * i + ' '
            text = text.replace(heading_marker, '')
        text = text.replace('#', '')

        # Remove other common markdown elements
        text = text.replace('===', '')
        text = text.replace('---', '')
        text = text.replace('```', '')
        text = text.replace('`', '')
        text = text.replace('> ', '')

        # Replace markdown list markers with plain text alternatives
        lines = text.split('\n')
        for i in range(len(lines)):
            # Replace bullet points with plain bullets
            if lines[i].strip().startswith('- '):
                lines[i] = lines[i].replace('- ', '• ', 1)
            # Handle numbered lists
            if len(lines[i]) > 2 and lines[i][0].isdigit() and lines[i][1] == '.' and lines[i][2] == ' ':
                lines[i] = '  ' + lines[i][3:]

        # Put it back together
        text = '\n'.join(lines)

        # Clean up any double spaces resulting from replacements
        while '  ' in text:
            text = text.replace('  ', ' ')

        # Clean up any extra newlines
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')

        return text

    def format_final_state(self, final_state) -> dict:
        """Format the final state with evaluation results."""
        enhanced_prompt = final_state["keys"].get("enhanced_prompt", "N/A")
        # Clean any remaining markdown
        enhanced_prompt = self.clean_markdown(enhanced_prompt)

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

            # Create a formatted plain text version of the results
            result = self.format_final_state(final_state)

            # Add original prompt answer and final prompt answer from nodes
            result["original_prompt_answer"] = final_state["keys"].get(
                "original_prompt_answer", "")
            result["original_prompt_lin_probs"] = final_state["keys"].get(
                "original_prompt_lin_probs", 0.0)
            result["final_prompt_answer"] = final_state["keys"].get(
                "final_prompt_answer", "")
            result["final_prompt_lin_probs"] = final_state["keys"].get(
                "final_prompt_lin_probs", 0.0)

            return result

        except Exception as e:
            print(colored(f"Error during workflow execution: {str(e)}",
                          'red', attrs=["bold"]))
            raise
