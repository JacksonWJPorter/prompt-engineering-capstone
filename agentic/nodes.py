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
        super().__init__(prompt, category="default-chat-openai", event_emitter=event_emitter)
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
        super().__init__(prompt, category="default-chat-openai", event_emitter=event_emitter)
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
            template="""**Role**: You are an LLM prompt evaluation expert with a precise scoring system that is outlined below.

**Score the prompt (1-99) based on these criteria:**
<scoring criteria>
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
</scoring criteria>

**BONUS POINTS FOR EXCELLENT PROMPTS:**
<bonus points>
- Exceptional clarity and precision: +1 to +5 points
- Perfect balance of detail and brevity: +1 to +4 points
- Clever use of structure for maximum clarity: +1 to +3 points
- Expert-level domain knowledge integration: +1 to +4 points
- Novel approach to a complex problem: +1 to +4 points
</bonus points>

**PENALTIES FOR VAGUE PROMPTS:**
<penalties>
- Prompt lacks specific objective: -8 to -15 points
- Missing necessary context: -5 to -15 points (depending on severity)
- No clarity on desired output format: -5 to -10 points
- Ambiguous terminology: -3 to -8 points per instance
- No scope limits or constraints: -5 to -12 points
- Generic requests like "tell me about X": maximum score of 40
</penalties>

**SPECIFIC EXAMPLE SCORES ACROSS THE FULL RANGE:**
<example scores>
- **Poor Prompts (Score 1–21):**
   - *Low-side example:* "Explain something."  
     *(Severely lacking in clarity, context, and structure; the request is too generic to guide any meaningful response.)*
   - *High-side example:* "What is love?"  
     *(A minimalistic question that provides virtually no actionable details or context, making it extremely ambiguous.)*

- **Extremely Vague Prompts (Score 22–35):**
   - *Low-side example:* "Talk about history."  
     *(No clear objective, lacks specific focus, and does not specify any time period, region, or aspect of history.)*
   - *High-side example:* "Discuss the evolution of technology over the decades."  
     *(Offers a general direction but remains overly broad without specifying which technologies or key aspects to address.)*

- **Weak Prompts with Ambiguity (Score 36–50):**
   - *Low-side example:* "Describe a process."  
     *(Ambiguous in nature as it fails to indicate which process to describe, the expected detail level, or any relevant context.)*
   - *High-side example:* "Explain how a computer works."  
     *(Provides some clarity on the subject matter but omits necessary constraints such as target audience, depth of explanation, or focus on specific components like hardware vs. software.)*

- **Subpar Prompts with Limited Direction (Score 51–60):**
   - *Low-side example:* "Outline a plan."  
     *(The objective is minimally identifiable, but the prompt lacks sufficient context, detailed guidelines, or explicit formatting instructions.)*
   - *High-side example:* "Provide a plan for a small business strategy."  
     *(Shows intent and direction with a recognizable topic, yet still misses detailed context, explicit formatting, and clear constraints.)*

- **Basic Clear Prompts (Score 61–68):**
   - *Low-side example:* "Write a summary of recent tech trends."  
     *(The objective is clear, but the prompt lacks depth in structure, measurable criteria, or any constraints to guide the response.)*
   - *High-side example:* "Summarize recent technology trends in a concise paragraph, focusing on two major innovations and their impact."  
     *(The prompt is clear and specific in its objective, but it does not require a highly detailed or structured output.)*

- **Competent Prompts (Score 69–75):**
   - *Low-side example:*  
     "Develop a basic outline for a research paper on renewable energy that includes at least four sections:
      - **Introduction:** Present the topic and its importance.
      - **Background:** Provide necessary context and historical data.
      - **Analysis:** Identify key areas of innovation or research.
      - **Conclusion:** Summarize insights.
      For each section, add one bullet point describing the intended focus.  
      *(This prompt demonstrates a clear structure and objective but lacks detailed constraints and advanced formatting.)*
      
   - *High-side example:*  
     "Create a structured outline for a research paper on renewable energy, ensuring to include five distinct sections:
      - **Introduction:** Define the research question and relevance.
      - **Literature Review:** Summarize key studies and theories.
      - **Methodology:** Describe the proposed research methods.
      - **Analysis:** Outline the data analysis approach.
      - **Conclusion:** Highlight expected outcomes and implications.
      Under each section, provide 2-3 bullet points that briefly capture the main ideas.  
      *(This prompt shows a balanced clarity and organization while including structured elements, yet it stops short of advanced formatting or explicit technical constraints.)*

- **Well-Formulated Prompts (Score 76–83):**
   - *Low-side example:*  
     "Draft a comprehensive report on current market trends with the following structure:
      - **Introduction:** Explain the scope and purpose.
      - **Data Analysis:** Present recent data with at least one table or simple chart.
      - **Key Findings:** List the major trends and insights.
      - **Conclusion:** Summarize the overall market direction.
      Include brief citations for data sources and use standard headings to separate sections.  
      *(This prompt is well-organized with clear sections and minimal formatting requirements but lacks additional contextual depth and advanced visual formatting.)*
      
   - *High-side example:*  
     "Produce an in-depth report on current market trends that must include:
      - **Introduction:** Clearly state the objectives, scope, and context for the market analysis.
      - **Data Analysis:** Provide a detailed examination of recent market data, including at least one visually rich element (chart or table) with a brief commentary.
      - **Market Drivers:** Identify and discuss the primary factors influencing the trends in a bullet list.
      - **Conclusion:** Offer a concise summary of insights and implications for future trends.
      Use markdown formatting with explicit headings, subheadings, and bullet points; cite all data sources in a standard format.  
      *(This prompt applies advanced structure and clear expectations while guiding the respondent to use specific formatting and citation practices.)*

- **High-Quality Prompts (Score 84–90):**
   - *Low-side example:*  
     "Design a robust project plan for a new mobile application that includes:
      - **Project Objectives:** Clearly define the goals and expected outcomes.
      - **Timeline:** Outline a detailed schedule with key milestones.
      - **Deliverables:** List specific, measurable outputs for each project phase.
      - **Risk Management:** Identify potential risks with brief mitigation strategies.
      - **Summary:** Provide a short overview of the overall strategy.
      Organize your plan using clear headings and bullet points.  
      *(This prompt is detailed and well-organized, reflecting a high-quality request, though it offers room for more granular instructions and advanced formatting.)*
      
   - *High-side example:*  
     "Develop an exhaustive project plan for a new mobile application that adheres to professional project management standards. Your plan must include:
      - **Project Objectives:** Define precise goals, key performance indicators (KPIs), and success criteria.
      - **Detailed Timeline:** Present a comprehensive timeline featuring milestones, deadlines, and task dependencies (incorporate a Gantt chart or equivalent if applicable).
      - **Deliverables:** Enumerate all deliverables with clear quality criteria and acceptance standards.
      - **Risk Management:** Provide a thorough risk assessment, including identification, impact analysis, and detailed mitigation strategies for each risk.
      - **Assumptions and Constraints:** Clearly list all relevant assumptions, constraints, and dependencies that could affect the project.
      - **Formatting and Structure:** Use markdown with explicit headings, subheadings, bullet points, and tables or diagrams where necessary to enhance clarity.
      - **Summary and Next Steps:** Conclude with a comprehensive summary of the overall strategy and propose actionable next steps.
      Cite any methodologies or frameworks used, ensuring that your plan is detailed, precise, and follows best prompt engineering practices.  
      *(This prompt sets a high bar with extensive requirements, advanced formatting, and clear constraints, making it very challenging to exceed without perfection.)*
  
- **Outstanding Prompts (Score 91–99):**
   - *Low-side example:*  
     "Compose a strategic blueprint for launching an innovative product that includes detailed sections on market analysis, competitive benchmarking, risk assessment, and execution steps; incorporate supporting data and citations as needed."  
     *(An outstanding prompt that is comprehensive and detailed, though it could be further refined with stricter formatting or more precise constraints.)*
   - *High-side example:*  
     ```
     Role: You are a sports data analyst tasked with providing accurate and verified statistics for a professional hockey player.

     User prompt: Please provide official, verified statistics for Connor McDavid's performance during the 2023 NHL regular season. Include only regular season statistics and no playoff data. The required metrics are:
     - Total goals scored
     - Total assists made
     - Total points (sum of goals and assists)
     - Source of the data

     Ensure the data is extracted from one of the following reputable sources: NHL.com, Hockey-Reference.com, or ESPN’s NHL stats database. Include the full URL of the exact page used as the source for verification.

     Also, briefly summarize McDavid’s overall performance that season in 1-2 sentences, highlighting key achievements or milestones if applicable.

     Format your response strictly in this JSON structure:
     ```json
     {{
       "player": "Connor McDavid",
       "season": "2023",
       "goals": <number>,
       "assists": <number>,
       "points": <number>,
       "summary": "",
       "source": ""
     }}
     ```
     Ensure numerical accuracy, citation credibility, and strict adherence to the JSON format.
     ```  
     *(This exemplary prompt provides exceptional clarity, comprehensive context, and intricate formatting requirements, reflecting best practices that are near the pinnacle of prompting excellence.)*
  
*Note: These examples are arbitrary and only serve as illustrative benchmarks; the topic does not influence the score.*
</example scores>

**IMPORTANT - Evaluate independently and use the FULL range to 99:**
<important notes>
- Think through your evaluation carefully and consider each criterion.
- DO NOT artificially cap scores at 90 or 95.
- Truly exceptional prompts CAN and SHOULD score 95-99.
- DO NOT inflate scores for average or subpar prompts.
- Use the full range to reflect the prompt's quality accurately.
</important notes>

**Your final score should reflect a detailed assessment:**
<final scoring notes>
- Score each criterion independently.
- Use precise point allocations (not just multiples of 5).
- Consider specific strengths and weaknesses.
- BE STRICT on prompts lacking specificity, context, or clear expectations.
- BE GENEROUS with truly excellent prompts, allowing scores of 95-99.
</final scoring notes>

**Return strictly in this JSON format:**
<JSON format>
{{
    "score": <integer_between_1_and_99>,
    "justification": "<One precise sentence highlighting critical flaws or excellence>",
    "improvement_suggestions": [
        "<specific improvement for highest priority gap>",
        "<specific improvement for second priority gap>",
        "<specific improvement for third priority gap>"
    ]
}}
</JSON format>

**PROMPT TO EVALUATE:**
<prompt>
{prompt}
</prompt>
        """,
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
            score += random.randint(-3, 0)
            score = min(score, 99)

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
