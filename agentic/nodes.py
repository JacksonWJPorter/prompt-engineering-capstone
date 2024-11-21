from typing import Dict, TypedDict, Any, TypeVar
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from termcolor import colored
from dataclasses import dataclass, field
from dotenv import load_dotenv
from utils.file import load_dictionary_agentic, load_safety_settings
import numpy as np

load_dotenv()

T = TypeVar('T', bound='GraphState')


@dataclass
class GraphState:
    keys: Dict[str, Any] = field(default_factory=dict)

    def update_keys(self, new_data: Dict[str, Any]) -> None:
        self.keys.update(new_data)

    @classmethod
    def create_initial_state(cls: type[T], initial_data: Dict[str, Any] = None) -> T:
        return cls(keys=initial_data or {})

    def get_value(self, key: str, default: Any = None) -> Any:
        return self.keys.get(key, default)


class BaseNode:
    def process(self, state: GraphState) -> GraphState:
        raise NotImplementedError("Subclasses must implement process()")

    def __call__(self, state: GraphState) -> GraphState:
        return self.process(state)


class OriginalPromptNode(BaseNode):
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=1,
            timeout=None,
            api_key='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',
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


class CallChatOpenAI(BaseNode):
    def __init__(self, prompt: str, category: str = "default-chat-openai"):
        self.prompt = prompt
        self.category = category
        self.model_configs = self.load_category_specific_config()
        self.model = self.init_model()
        self.output_parser = StrOutputParser()

    def load_category_specific_config(self) -> Dict[str, Any]:
        try:
            # return load_dictionary_agentic(self.category)
            return load_dictionary_agentic('default-chat-openai')
        except Exception as e:
            print(f"Error loading config for category '{self.category}': {e}")
            return {}

    def init_model(self, model_configs={}) -> ChatOpenAI:
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


class CategorizePromptNode(CallChatOpenAI):
    def __init__(self, prompt: str):
        super().__init__(prompt, category="default-chat-openai")
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


class RephraseNode(CallChatOpenAI):
    def __init__(self, prompt: str):
        super().__init__(prompt, category="default-chat-openai")
        self.prompt_template = PromptTemplate(
            template="""
            # Role: You are an expert at helping users get the best output from LLMs.
            
            #Task: Analyze the user's question and rephrase it into a concise and effective prompt for a large language model. 
            
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
        prompt = state.get_value("original_prompt", self.prompt)

        rephrased_question = self.call_chat_openai(
            self.prompt_template,
            {"question": prompt}
        ).strip()

        print(colored("Rephrased Prompt: ", 'light_magenta',
              attrs=["bold"]), rephrased_question)
        state.update_keys({"rephrased_question": rephrased_question})
        return state


class PromptEnhancerNode(CallChatOpenAI):
    def __init__(self, prompt: str):
        super().__init__(prompt)
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


class QualityCheckNode(CallChatOpenAI):
    def __init__(self, prompt: str, quality_criteria: list):
        super().__init__(prompt, category="default-chat-openai")
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


class FinalAnswerNode(BaseNode):
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            timeout=None,
            api_key='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',
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
