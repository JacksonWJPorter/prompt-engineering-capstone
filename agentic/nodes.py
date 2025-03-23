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
            "node_output": None,
            "change_text": f"Processed {self.__class__.__name__}"
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
            ## Think step-by-step to make sure you get the answer correct.
            </task instructions>

            ## Exemplars of a user query and target chosen category:
            <exemplar outputs with target formatting>
            {{query="What is the capital of France?", category="Simple Question"}},
            {{query="Summarize the main points of the French Revolution.", category="Summary"}},
            {{query="Write a short story about a young woman who travels through time to meet
            Marie Antoinette.", category="Creative Writing and Ideation"}},
            {{query="My French drain is overflowing, how do I troubleshoot this problem?",
            category="Problem Solving"}},
            {{query="Translate 'Hello, how are you?' into French.", category="Simple Question"}},
            {{query="Give me a list of all the kings of France.", category="Simple Question"}},
            {{query="I need ideas for a French-themed birthday party.",
            category="Creative Writing and Ideation"}},
            {{query="What are the best French restaurants in Paris?",
            category="Simple Question"}},
            {{query="Explain the rules of French grammar.", category="Summary"}},
            {{query="My car's 'check engine' light is on and the code reader says it's a P0420
            error. What should I do?", category="Problem Solving"}},
            {{query="Write me a Python script to detect a palindrome", category="Coding and Programming"}},
            </exemplar outputs with target formatting>

            # User query to categorize:
            {question}

            # IMPORTANT: YOUR OUTPUT MUST BE EXACTLY ONE OF THE FOLLOWING WITH NO OTHER WORDS ATTACHED:
            <output options>
                Simple Question
                Summary
                Creative Writing and Ideation
                Problem Solving
                Coding and Programming
                Other
            </output options>
            """,
            input_variables=["question"]
        )

    def process(self, state: GraphState) -> GraphState:
        prompt = state.get_value("original_prompt", self.prompt)
        category = self.call_chat_openai(
            self.prompt_template, {"question": prompt}).strip()

        print(colored("Category Determined: ",
              'light_magenta', attrs=["bold"]), category)
        state.update_keys({"category": category})
        return state

    def get_node_data(self, state: GraphState) -> dict:
        return {
            "node_type": "CategorizePromptNode",
            "node_output": state.get_value("category", "unknown"),
            "change_text": f"Prompt categorized as: {state.get_value('category', 'unknown')}"
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
            "node_output": state.get_value("rephrased_question", ""),
            "change_text": "Prompt rephrased for better AI understanding"
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
            (1) Assign a specific role to the LLM that will be prompted, corresponding
            to the task the user needs completed.
            (2) Improve formatting of the user prompt to make it easier for the
            LLM to understand using plain text formatting only.
            (3) Fix any typos.
            (4) Include relevant details from the context.
            (5) DO NOT use any markdown formatting in your response. No asterisks, no hashtags, 
               no backticks, no formatting symbols of any kind.
            (6) Use plain text only with regular paragraphs, line breaks, and simple formatting 
               like dashes or bullets using standard characters.
            (7) Do not use ALL CAPS for emphasis. Use normal sentence case throughout.
            (8) For structure, use line breaks, indentation, and plain text bullet points (•).

            --- [User prompt to improve]: {user_prompt}

            IMPORTANT: YOUR RESPONSE MUST BE 100% PLAIN TEXT WITH ABSOLUTELY NO MARKDOWN 
            FORMATTING SYMBOLS OF ANY KIND AND NO TEXT IN ALL CAPS.
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

        # Clean up any markdown and normalize capitalization
        enhanced_prompt = self.clean_formatting(enhanced_prompt)

        print(colored("Enhanced Prompt: ", 'light_magenta',
              attrs=["bold"]), enhanced_prompt)
        state.update_keys({"enhanced_prompt": enhanced_prompt})
        return state

    def clean_formatting(self, text):
        """Remove all markdown formatting and normalize capitalization."""
        if not text:
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

        # Normalize capitalization - find words in ALL CAPS and convert to normal case
        # This preserves normal capitalization while fixing ALL CAPS words
        words = text.split()
        for i in range(len(words)):
            # Check if word is all uppercase and longer than 1 character
            if words[i].isupper() and len(words[i]) > 1:
                # Convert to title case if it's a proper noun, lowercase otherwise
                words[i] = words[i].title()

        # Rejoin with spaces
        text = ' '.join(words)

        # Clean up any double spaces resulting from replacements
        while '  ' in text:
            text = text.replace('  ', ' ')

        # Clean up any extra newlines
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')

        return text

    def get_node_data(self, state: GraphState) -> dict:
        return {
            "node_type": "PromptEnhancerNode",
            "node_output": state.get_value("enhanced_prompt", ""),
            "change_text": "Prompt enhanced with additional context and structure"
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
        - Vague/generic prompt (e.g., "Tell me about AI", "Write an essay") → 30-50 range
        - List or recommendation request (e.g., "What are the safest countries for solo travelers?") → 50-65 range 
        - Simple task with minimal context → 46-70 range
        - Creative writing assistance (e.g., "Help developing a character for my novel") → 65-80 range
        - Moderate complexity task with some context → 65-90 range
        - Complex task with detailed requirements → 80-99 range

        Score the prompt (1-99) based on these criteria:
        1. Specificity (40 points):
           - Core clarity: Clear and precise objective (15 points)
           - Required context (based on prompt needs): (15 points)
           - Parameters and constraints (10 points)

        2. Structure & Clarity (30 points):
           - Clear phrasing (10 points)
           - Logical organization (10 points)
           - Professional tone (10 points)

        3. Response Expectations (30 points):
           - Detail requirements (10 points)
           - Format specifications (10 points)
           - Quality standards (10 points)

        BONUS POINTS FOR EXCELLENT PROMPTS:
        - Exceptional clarity and precision: +1 to +5 points
        - Perfect balance of detail and brevity: +1 to +4 points
        - Clever use of structure for maximum clarity: +1 to +3 points
        - Expert-level domain knowledge integration: +1 to +4 points
        - Novel approach to a complex problem: +1 to +4 points

        PENALTIES FOR VAGUE PROMPTS:
        - Prompt lacks specific objective: -8 to -15 points
        - Missing necessary context: -5 to -15 points (depending on severity)
        - No clarity on desired output format: -5 to -10 points
        - Ambiguous terminology: -3 to -8 points per instance
        - No scope limits or constraints: -5 to -12 points
        - Generic requests like "tell me about X": maximum score of 40

        CRITICAL SCORING INSTRUCTIONS:
        1. Use ANY END DIGIT in your scoring (0-9):
           - Feel free to use scores ending in any digit, including 0 and 5
           - Don't artificially avoid certain end digits
           - Choose the most accurate score regardless of the end digit
        
        2. SPECIFIC EXAMPLE SCORES ACROSS THE FULL RANGE:
           - For vague prompts: 22, 27, 31, 34, 38, 42 (lower scores for greater vagueness)
           - For basic questions: 32, 37, 41, 44, 48 (not always 40 or 45)
           - For list/recommendation requests: 52, 54, 57, 59, 63 (varied in 50-65 range)
           - For simple tasks: 53, 58, 62, 66, 69 (not just multiples of 5)
           - For creative writing prompts: 66, 71, 74, 76, 78, 79 (varied in appropriate range)
           - For standard well-formed prompts: 67, 72, 78, 83, 87, 89 (varied, not just rounded numbers)
           - For enhanced prompts: 73, 77, 82, 84, 86, 88, 91, 93 (NOT capped at 90)
           - For excellent prompts: 87, 89, 91, 93, 95, 96, 97, 98, 99 (use the FULL range up to 99)
        
        3. IMPORTANT - Evaluate independently and use the FULL range to 99:
           - DO NOT artificially cap scores at 90 or 95
           - Truly exceptional prompts CAN and SHOULD score 95-99
           - Minor improvements might add 2-5 points
           - Moderate improvements might add 6-12 points
           - Major improvements might add 13-20 points
           - Some poor "enhancements" might even reduce the score

        4. Your final score should reflect a detailed assessment:
           - Score each criterion independently
           - Use precise point allocations (not just multiples of 5)
           - Consider specific strengths and weaknesses
           - BE STRICT on prompts lacking specificity, context, or clear expectations
           - BE GENEROUS with truly excellent prompts, allowing scores of 95-99

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
                    "score": 50,  # Could be any default score
                    "justification": "Error during evaluation",
                    "suggestions": []
                }
            })

        return state

    def get_node_data(self, state: GraphState) -> dict:
        return {
            "node_type": "PromptEvaluationNode",
            "node_output": state.get_value("prompt_evaluation", {}),
            "change_text": f"Prompt evaluated with score: {state.get_value('prompt_evaluation', {})}/100"
        }
