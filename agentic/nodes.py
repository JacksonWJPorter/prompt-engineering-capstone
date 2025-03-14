import json
from dataclasses import dataclass
from typing import Dict, TypedDict, Any, TypeVar
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from termcolor import colored
from dataclasses import dataclass, field
from dotenv import load_dotenv
from utils.file import load_dictionary_agentic, load_safety_settings
import numpy as np
import os
import random

load_dotenv()

T = TypeVar('T', bound='GraphState')


@dataclass
class GraphState:
    keys: Dict[str, Any] = field(default_factory=dict)
    clarification_history: list = field(default_factory=list)

    def update_keys(self, new_data: Dict[str, Any]) -> None:
        self.keys.update(new_data)

    def add_clarification(self, question: str, answer: str) -> None:
        self.clarification_history.append({
            "question": question,
            "answer": answer
        })

    def get_clarification_history_text(self) -> str:
        history_text = f"###Original Prompt###: {self.keys.get('original_prompt', '')}\n"
        for item in self.clarification_history:
            history_text += f"---Clarification Question: {item['question']}\n"
            history_text += f"---User Response: {item['answer']}\n"

        return history_text

    @classmethod
    def create_initial_state(cls: type[T], initial_data: Dict[str, Any] = None) -> T:
        return cls(keys=initial_data or {}, clarification_history=[])

    def get_value(self, key: str, default: Any = None) -> Any:
        return self.keys.get(key, default)


class BaseNode:
    def __init__(self, event_emitter=None):
        self.event_emitter = event_emitter

    def process(self, state: GraphState) -> GraphState:
        raise NotImplementedError("Subclasses must implement process()")

    def __call__(self, state: GraphState) -> GraphState:
        result_state = self.process(state)

        # Emit node completion event if an emitter is available
        if self.event_emitter:
            # Get node name from class name
            node_name = self.__class__.__name__

            # Get relevant data to send to frontend
            node_data = self.get_node_data(result_state)

            # Emit the event
            self.event_emitter.emit_node_completion(node_name, node_data)

        return result_state

    def get_node_data(self, state: GraphState) -> dict:
        """
        Extract relevant data from the state to send to the frontend.
        Override this in specific node classes to customize data.
        """
        # Default implementation returns node type and output
        return {
            "node_type": self.__class__.__name__,
            "node_output": None  # Default to None, should be overridden by subclasses
        }

    def __init__(self, event_emitter=None):
        self.event_emitter = event_emitter

    def process(self, state: GraphState) -> GraphState:
        raise NotImplementedError("Subclasses must implement process()")

    def __call__(self, state: GraphState) -> GraphState:
        result_state = self.process(state)

        # Emit node completion event if an emitter is available
        if self.event_emitter:
            # Get node name from class name
            node_name = self.__class__.__name__

            # Get relevant data to send to frontend
            node_data = self.get_node_data(result_state)

            # Emit the event
            self.event_emitter.emit_node_completion(node_name, node_data)

        return result_state

    def get_node_data(self, state: GraphState) -> dict:
        """
        Extract relevant data from the state to send to the frontend.
        Override this in specific node classes to customize data.
        """
        # Default implementation returns basic info
        return {
            "status": "completed",
            "node_type": self.__class__.__name__,
            "current_step": state.get_value("current_step", "unknown")
        }


class OriginalPromptNode(BaseNode):
    def __init__(self, prompt: str, event_emitter=None):
        super().__init__(event_emitter)
        self.prompt = prompt
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=1,
            timeout=None,
            api_key=os.getenv("OPENAI_API_KEY"),
            logprobs=True
        )

    def process(self, state: GraphState) -> GraphState:
        original_prompt = state.get_value("original_prompt", self.prompt)
        try:
            response = self.model.invoke(original_prompt)
            logprobs_result = response.response_metadata['logprobs']['content']
            logprobs = [item['logprob'] for item in logprobs_result]
            avg_lin_probs_initial = np.exp(float(np.mean(logprobs)))

            print(colored("Original Prompt: ", 'blue',
                  attrs=["bold"]), original_prompt)
            print(colored("Original Answer: ",
                  'blue', attrs=["bold"]), response.content)
            print(colored("Original Lin Probs: ",
                  'blue', attrs=["bold"]), avg_lin_probs_initial)

            state.update_keys({
                "original_prompt_answer": response.content,
                "original_prompt_lin_probs": avg_lin_probs_initial
            })
            return state
        except Exception as e:
            print(f"Error in OriginalPromptNode: {e}")
            return state

    def get_node_data(self, state: GraphState) -> dict:
        return {
            "node_type": "OriginalPromptNode",
            "node_output": state.get_value("original_prompt_answer", "")
        }


class CallChatOpenAI(BaseNode):
    def __init__(self, prompt: str, category: str = "default-chat-openai", event_emitter=None):
        super().__init__(event_emitter)
        self.prompt = prompt
        self.category = category
        self.model_configs = self.load_category_specific_config()
        self.model = self.init_model()
        self.output_parser = StrOutputParser()

    def load_category_specific_config(self) -> Dict[str, Any]:
        try:
            # Load configuration based on the given category
            return load_dictionary_agentic(self.category)
        except Exception as e:
            print(f"Error loading config for category '{self.category}': {e}")
            return {}

    def init_model(self, model_configs: Dict[str, Any] = None) -> ChatOpenAI:
        try:
            model_configs = model_configs or self.model_configs
            return ChatOpenAI(**model_configs)
        except Exception as e:
            print(f"Error initializing ChatOpenAI model: {e}")
            return None

    def call_chat_openai(self, prompt_template: PromptTemplate, variables: Dict[str, Any]) -> str:
        if not prompt_template:
            raise ValueError("A prompt template must be provided.")
        try:
            chain = prompt_template | self.model | self.output_parser
            response = chain.invoke(variables).strip()
            return response
        except Exception as e:
            print(f"Error invoking ChatOpenAI model: {e}")
            return "Error: Model invocation failed."

    def process(self, state: GraphState) -> GraphState:
        raise NotImplementedError("Subclasses must implement process()")

    def get_node_data(self, state: GraphState) -> dict:
        return {
            "node_type": self.__class__.__name__,
            "prompt": self.prompt
        }


class CategorizePromptNode(CallChatOpenAI):
    def __init__(self, prompt: str, event_emitter=None):
        super().__init__(prompt, category="default-chat-openai", event_emitter=event_emitter)
        self.prompt_template = PromptTemplate(
            template="""
            # Role: You are an AI prompt analyzer tasked with identifying the specific category
            of question, from semi-generic prompts inputted by your user.

            ## Task:
            **Please review user query and determine the category.**

            # Task Instructions:
            <task instructions>
            ## Review the user prompt carefully and assign ONE category.
            ## Please use only the following categories: 'simple question', 'summary',
            'creative writing and ideation', 'problem solving', 'other'
            ## Output ONE of the categories from the list provided to you.
            ## Think step-by-step to make sure you get the answer correct.
            </task instructions>

            ## Exemplars of a user query and target chosen category:
            <exemplar outputs with target formatting>
            {{query="What is the capital of France?", category="simple question"}},
            {{query="Summarize the main points of the French Revolution.", category="summary"}},
            {{query="Write a short story about a young woman who travels through time to meet
            Marie Antoinette.", category="creative writing and ideation"}},
            {{query="My French drain is overflowing, how do I troubleshoot this problem?",
            category="problem solving"}},
            {{query="Translate 'Hello, how are you?' into French.", category="simple question"}},
            {{query="Give me a list of all the kings of France.", category="simple question"}},
            {{query="I need ideas for a French-themed birthday party.",
            category="creative writing and ideation"}},
            {{query="What are the best French restaurants in Paris?",
            category="simple question"}},
            {{query="Explain the rules of French grammar.", category="summary"}},
            {{query="My car's 'check engine' light is on and the code reader says it's a P0420
            error. What should I do?", category="problem solving"}}
            </exemplar outputs with target formatting>

            # User query to categorize:
            {question}

            # IMPORTANT: YOUR OUTPUT MUST BE EXACTLY ONE OF THE FOLLOWING:
            simple-question, summary, creative-writing-and-ideation, problem-solving, other
            """,
            input_variables=["question"]
        )

    def process(self, state: GraphState) -> GraphState:
        prompt = state.get_value("original_prompt", self.prompt)
        category = self.call_chat_openai(
            self.prompt_template, {"question": prompt}).strip()

        category_map = {
            "simple question": 'simple-question',
            "summary": 'summary',
            "creative writing and ideation": 'creative-writing-and-ideation',
            "problem solving": 'problem-solving',
            "other": 'other'
        }

        if category in category_map:
            category = category_map[category]

        print(colored("Category Determined: ",
              'light_magenta', attrs=["bold"]), category)
        state.update_keys({"category": category})
        return state

    def get_node_data(self, state: GraphState) -> dict:
        return {
            "node_type": "CategorizePromptNode",
            "node_output": state.get_value("category", "unknown")
        }


class QueryDisambiguationNode(CallChatOpenAI):
    def __init__(self, prompt: str, event_emitter=None):
        super().__init__(prompt, category="query-disambiguation", event_emitter=event_emitter)
        self.prompt_template = PromptTemplate(
            template="""
            # Role: You are a clarification assistant.
            # Task: Review the conversation history and current query to determine if further clarification is needed.
            
            # Context:
            {history}
            
            # Important: Your response should either be "clear" if the query and its context are clear, 
            or if anything remains unclear, your response should be a specific question to resolve the ambiguity.
            Do not repeat previous clarification questions.
            
            **Important**: The query does not need to be perfectly clear in all aspects down to every specific.
            If it is generally clear enough, mark it as "clear".
            
            ## Output:
            """,
            input_variables=["history"]
        )

    def process(self, state: GraphState) -> GraphState:
        # Get full conversation history
        history = state.get_clarification_history_text()

        clarification_question = self.call_chat_openai(
            self.prompt_template,
            {"history": history}
        ).strip().lower()

        print(colored("Clarification Question: ", 'cyan',
              attrs=["bold"]), clarification_question)
        state.update_keys({"clarification_question": clarification_question})
        return state

    def get_node_data(self, state: GraphState) -> dict:
        return {
            "node_type": "QueryDisambiguationNode",
            "node_output": state.get_value("clarification_question", "")
        }


class RephraseNode(CallChatOpenAI):
    def __init__(self, prompt: str, event_emitter=None):
        super().__init__(prompt, category="default-chat-openai", event_emitter=event_emitter)
        self.prompt_template = PromptTemplate(
            template="""
            # Role: You are an expert at helping users get the best output from LLMs.

            #Task: Analyze the user's question and rephrase it into a concise and effective prompt for a large language model.
            If the question has a history of clarifications of the original question, rephrase it all into one clear question and add the context
            from the various clarifications.

            #Task instructions:
            1. Clearly state the desired output: Specify what information or task the model should perform.
            2. Provides context and background information: Include relevant details from the
            context to guide the model's understanding.
            3. Uses clear and concise language: Avoid ambiguity and use easily understood language.
            4. Is formatted for optimal processing: Use appropriate markup, formatting, or
            techniques to enhance readability and processing efficiency.

            # User prompt:
            {question}

            # Rephrased prompt:
            """,
            input_variables=["question"]
        )

    def process(self, state: GraphState) -> GraphState:
        prompt = state.get_value("human_feedback") or state.get_value(
            "original_prompt", self.prompt)
        rephrased_question = self.call_chat_openai(
            self.prompt_template,
            {"question": prompt}
        ).strip()

        print(colored("Rephrased Prompt: ", 'light_magenta',
              attrs=["bold"]), rephrased_question)
        state.update_keys({"rephrased_question": rephrased_question})
        return state

    def get_node_data(self, state: GraphState) -> dict:
        return {
            "node_type": "RephraseNode",
            "node_output": state.get_value("rephrased_question", "")
        }


class PromptEnhancerNode(CallChatOpenAI):
    def __init__(self, prompt: str, event_emitter=None):
        super().__init__(prompt, category="default-chat-openai", event_emitter=event_emitter)
        self.prompt_template = PromptTemplate(
            template="""
            # Role: You are an expert prompt enhancer who is given a suboptimal prompt.

            # Your Task: Enhance the prompt provided by the user to obtain the best response
            from an LLM that will be prompted with it.
            You do this based on the category of the prompt and its corresponding instructions.

            #[Prompt Category]: {category}
            #[Instructions]:
            {specific_template}

            # Always follow these guidelines:
            (1) Assign a highly specific role to the LLM that will be prompted, corresponding
            to the task the user needs completed.
            (2) Add markup and improve formatting of the user prompt to make it easier for the
            LLM to understand.
            (3) Fix any typos.
            (4) Include relevant details from the context.

            --- [User prompt to improve]: {user_prompt}

            IMPORTANT: YOUR RESPONSE TO ME SHOULD BE AN ENHANCED VERSION OF THE
            [User prompt to improve].
            """,
            input_variables=["category",
                             "user_prompt", "specific_template"]
        )

    def process(self, state: GraphState) -> GraphState:
        category = state.get_value("category", "other")
        user_prompt = state.get_value("rephrased_question", self.prompt)

        # params = load_dictionary_agentic(category)
        params = load_dictionary_agentic('default-chat-openai')
        specific_template = params.get('PROMPT', "No specific template found")

        enhanced_prompt = self.call_chat_openai(
            self.prompt_template,
            {
                "category": category,
                "user_prompt": user_prompt,
                "specific_template": specific_template
            },
        ).strip()

        print(colored("Enhanced Prompt: ", 'light_magenta',
              attrs=["bold"]), enhanced_prompt)
        state.update_keys({"enhanced_prompt": enhanced_prompt})
        return state

    def get_node_data(self, state: GraphState) -> dict:
        return {
            "node_type": "PromptEnhancerNode",
            "node_output": state.get_value("enhanced_prompt", "")
        }


class QueryDisambiguationNode(CallChatOpenAI):
    def __init__(self, prompt: str, event_emitter=None):
        super().__init__(prompt, category="query-disambiguation", event_emitter=event_emitter)
        self.prompt_template = PromptTemplate(
            template="""
            # Role: You are a clarification assistant.
            # Task: Review the conversation history and current query to determine if further clarification is needed.
            
            # Context:
            {history}
            
            # Important: Your response should either be "clear" if the query and its context are clear, 
            or if anything remains unclear, your response should be a specific question to resolve the ambiguity.
            Do not repeat previous clarification questions.
            
            **Important**: The query does not need to be perfectly clear in all aspects down to every specific.
            If it is generally clear enough, mark it as "clear".
            
            ## Output:
            """,
            input_variables=["history"]
        )

    def process(self, state: GraphState) -> GraphState:
        # Get full conversation history
        history = state.get_clarification_history_text()

        clarification_question = self.call_chat_openai(
            self.prompt_template,
            {"history": history}
        ).strip().lower()

        print(colored("Clarification Question: ", 'cyan',
              attrs=["bold"]), clarification_question)
        state.update_keys({"clarification_question": clarification_question})
        return state

    def get_node_data(self, state: GraphState) -> dict:
        return {
            "status": "completed",
            "node_type": "QueryDisambiguationNode",
            "clarification_question": state.get_value("clarification_question", ""),
            "needs_clarification": state.get_value("clarification_question", "") != "clear"
        }


class QualityCheckNode(CallChatOpenAI):
    def __init__(self, prompt: str, quality_criteria: list, event_emitter=None):
        super().__init__(prompt, category="default-chat-openai", event_emitter=event_emitter)
        self.quality_criteria = quality_criteria

    def process(self, state: GraphState) -> GraphState:
        enhanced_prompt = state.get_value("enhanced_prompt", self.prompt)
        if not enhanced_prompt:
            print("Enhanced prompt not found in state. Skipping quality check.")
            return state

        issues = []
        for criterion in self.quality_criteria:
            check_template = PromptTemplate(
                template="""
                Assess the {criterion} of this prompt: {question}
                """,
                input_variables=["criterion", "question"]
            )

            result = self.call_chat_openai(
                check_template,
                {"criterion": criterion, "question": enhanced_prompt}
            ).strip()

            if result == "Issue detected":
                issues.append(criterion)

        print(colored("Quality Issues: ",
              'light_magenta', attrs=["bold"]), issues)
        state.update_keys({"quality_issues": issues})
        return state

    def get_node_data(self, state: GraphState) -> dict:
        return {
            "status": "completed",
            "node_type": "QualityCheckNode",
            "enhanced_prompt": state.get_value("enhanced_prompt", ""),
            "quality_issues": state.get_value("quality_issues", [])
        }


class HumanNode(BaseNode):
    def __init__(self, prompt: str, event_emitter=None):
        super().__init__(event_emitter)
        self.prompt = prompt
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            timeout=None,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def process(self, state: GraphState) -> GraphState:
        print("\n" + "="*50)
        print(colored("Human Review Required", 'yellow', attrs=['bold']))

        # Get the latest clarification question
        clarification_question = state.get_value("clarification_question", "")
        if clarification_question == "clear":
            return state

        # Get human feedback
        lines = []
        while True:
            line = input()
            if line:
                lines.append(line)
            elif lines:
                break
            else:
                print("Please provide some feedback before continuing:")

        feedback = "\n".join(lines)

        # Add this Q&A pair to the history
        state.add_clarification(clarification_question, feedback)

        # Update the human_feedback key with the complete history
        state.update_keys(
            {"human_feedback": state.get_clarification_history_text()})

        print("Prompt with History:\n", state.keys['human_feedback'])
        print("\n" + "="*50)
        return state

    def get_node_data(self, state: GraphState) -> dict:
        return {
            "node_type": "HumanNode",
            "node_output": state.get_value("human_feedback", "")
        }


class FinalAnswerNode(BaseNode):
    def __init__(self, prompt: str, event_emitter=None):
        super().__init__(event_emitter)
        self.prompt = prompt
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            timeout=None,
            api_key=os.getenv("OPENAI_API_KEY"),
            logprobs=True
        )
        self.output_parser = StrOutputParser()

    def process(self, state: GraphState) -> GraphState:
        enhanced_prompt = state.get_value("enhanced_prompt", self.prompt)
        try:
            response = self.model.invoke(enhanced_prompt)
            logprobs_result = response.response_metadata['logprobs']['content']
            logprobs = [item['logprob'] for item in logprobs_result]
            avg_lin_probs_final = np.exp(float(np.mean(logprobs)))

            print(colored("Final Prompt: ", 'blue',
                  attrs=["bold"]), enhanced_prompt)
            print(colored("Final Answer: ", 'blue',
                  attrs=["bold"]), response.content)
            print(colored("Final Lin Probs: ", 'blue',
                  attrs=["bold"]), avg_lin_probs_final)

            state.update_keys({
                "final_prompt_answer": response.content,
                "final_prompt_lin_probs": avg_lin_probs_final,
            })
            return state
        except Exception as e:
            print(f"Error in FinalAnswerNode: {e}")
            return state

    def get_node_data(self, state: GraphState) -> dict:
        return {
            "node_type": "FinalAnswerNode",
            "node_output": state.get_value("final_prompt_answer", "")
        }


class VersioningNode(BaseNode):
    def __init__(self, step_name: str, event_emitter=None):
        super().__init__(event_emitter)
        self.step_name = step_name

    def process(self, state: GraphState) -> GraphState:
        # Get or initialize the steps list
        steps = state.get_value("workflow_steps", [])

        # Add the current step
        steps.append(self.step_name)

        # Update state with new steps
        state.update_keys({
            "workflow_steps": steps,
            "current_step": self.step_name
        })

        # Print progress
        print(colored("\n=== Workflow Progress ===", 'green', attrs=["bold"]))
        for idx, step in enumerate(steps, 1):
            print(colored(f"Step {idx}: {step}", 'green'))
        print(colored("=====================\n", 'green', attrs=["bold"]))

        return state

    def get_node_data(self, state: GraphState) -> dict:
        return {
            "node_type": "VersioningNode",
            "node_output": self.step_name
        }


class PromptEvaluationNode(CallChatOpenAI):
    def __init__(self, prompt: str, event_emitter=None):
        super().__init__(prompt, category="default-chat-openai", event_emitter=event_emitter)
        self.evaluation_template = PromptTemplate(
            template="""You are a demanding prompt evaluation expert with a precise scoring system.

        PROMPT TO EVALUATE:
        {prompt}

        First identify prompt complexity and appropriate score ranges:
        - Basic factual question (e.g., "Who is the president?") → 30-45 range
        - List or recommendation request (e.g., "What are the safest countries for solo travelers?") → 50-60 range 
        - Simple task with minimal context → 46-65 range
        - Creative writing assistance (e.g., "Help developing a character for my novel") → 70-79 range
        - Moderate complexity task with some context → 66-85 range
        - Complex task with detailed requirements → 86-98 range

        Score the prompt (1-99) based on these criteria:
        1. Specificity (40 points):
           - Core clarity: Clear and precise objective (10 points)
           - Required context (based on prompt needs): (20 points)
           - Parameters and constraints (10 points)

        2. Structure & Clarity (30 points):
           - Clear phrasing (10 points)
           - Logical organization (10 points)
           - Professional tone (10 points)

        3. Response Expectations (30 points):
           - Detail requirements (10 points)
           - Format specifications (10 points)
           - Quality standards (10 points)

        CRITICAL SCORING INSTRUCTIONS:
        1. Use VARIED END DIGITS in your scoring. Your scores should have a mix of end digits:
           - Use scores ending in 1, 2, 3, 4, 6, 7, 8, 9 MOST of the time
           - Occasionally use scores ending in 0 or 5, but not as your default
        
        2. SPECIFIC EXAMPLE SCORES:
           - For basic questions: 32, 37, 41, 44 (not always 40 or 45)
           - For list/recommendation requests: 52, 54, 57, 59 (within 50-60 range)
           - For simple tasks: 53, 58, 62, 64 (not always 50, 55, 60, 65)
           - For creative writing prompts: 71, 74, 76, 78 (within 70-79 range)
           - For standard prompts: 67, 71, 76, 83 (varied, not just 70, 75, 80, 85)
           - For enhanced prompts: 73, 77, 82, 84, 86 (NOT automatically 85)
        
        3. IMPORTANT - Each prompt must be evaluated independently:
           - DO NOT automatically add 10 points to rephrased/enhanced prompts
           - Some enhancements deserve +3 points, others +7, others +15, etc.
           - Minor improvements might add 2-5 points
           - Moderate improvements might add 6-12 points
           - Major improvements might add 13-20 points
           - Some poor "enhancements" might even reduce the score

        4. Your final score should reflect a detailed assessment:
           - Score each criterion independently
           - Use precise point allocations (not just multiples of 5)
           - Consider specific strengths and weaknesses

        Return strictly in this JSON format:
        {{
            "score": <integer_between_1_and_99>,
            "justification": "<One precise sentence highlighting critical flaws or excellence>",
            "improvement_suggestions": [
                "<specific improvement for highest priority gap>",
                "<specific improvement for second priority gap>",
                "<specific improvement for third priority gap>"
            ]
        }}""",
            input_variables=["prompt"]
        )

    def process(self, state: GraphState) -> GraphState:
        # Use original prompt exclusively for evaluation
        prompt = state.get_value("original_prompt", self.prompt)
        
        try:
            evaluation_json = self.call_chat_openai(
                self.evaluation_template,
                {"prompt": prompt}
            )

            # Clean JSON response
            evaluation_json = evaluation_json.strip()
            start_idx = evaluation_json.find("{")
            end_idx = evaluation_json.rfind("}")
            if start_idx >= 0 and end_idx >= 0 and end_idx > start_idx:
                evaluation_json = evaluation_json[start_idx:end_idx+1]

            results = json.loads(evaluation_json)
            score = int(results["score"])
            
            # If the score ends in 0 or 5, adjust it slightly
            if score % 5 == 0:
                # Add a small random adjustment (-2 to +2)
                adjustment = random.choice([-2, -1, 1, 2])
                score = max(1, min(99, score + adjustment))  # Keep between 1-99

            # Color coding based on score
            score_color = 'red'
            if score >= 90:
                score_color = 'green'
            elif score >= 70:
                score_color = 'yellow'
            elif score >= 50:
                score_color = 'magenta'

            # Print formatted evaluation with only score colored
            print("\n=== PROMPT EVALUATION ===")
            print(f"Score: {colored(str(score), score_color)}")
            print(f"Score Justification: {results['justification']}")
            print("\nImprovement Suggestions:")
            for i, suggestion in enumerate(results['improvement_suggestions'], 1):
                print(f"{i}. {suggestion}")

            # Update state
            state.update_keys({
                "prompt_evaluation": {
                    "score": score,
                    "justification": results["justification"],
                    "suggestions": results["improvement_suggestions"]
                }
            })

        except Exception as e:
            print(colored(f"Error in evaluation: {e}", "red"))
            state.update_keys({
                "prompt_evaluation": {
                    "score": 43,  # Using a non-multiple of 5 for errors too
                    "justification": "Error during evaluation",
                    "suggestions": []
                }
            })

        return state

    def get_node_data(self, state: GraphState) -> dict:
        return {
            "node_type": "PromptEvaluationNode",
            "node_output": state.get_value("prompt_evaluation", {})
        }
