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


class QueryDisambiguationNode(CallChatOpenAI):
    def __init__(self, prompt: str):
        super().__init__(prompt, category="query-disambiguation")
        self.prompt_template = PromptTemplate(
            template="""
            # Role: You are a clarification assistant.
            # Task: Review the conversation history and current query to determine if further clarification is needed.
            
            # Context:
            {history}
            
            # Important: Your response should either be "clear" if the query and its context are clear, 
            or if anything remains unclear, your response should be a specific question to resolve the ambiguity.
            Do not repeat previous clarification questions.
            
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


class RephraseNode(CallChatOpenAI):
    def __init__(self, prompt: str):
        super().__init__(prompt, category="default-chat-openai")
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


class HumanNode(BaseNode):
    def __init__(self, prompt: str):
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


class FinalAnswerNode(BaseNode):
    def __init__(self, prompt: str):
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


class VersioningNode(BaseNode):
    def __init__(self, step_name: str):
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


# ... (existing imports) ...
from dataclasses import dataclass
import json
from termcolor import colored

# Add this new class after your existing node classes
class PromptEvaluationNode(CallChatOpenAI):
    def __init__(self, prompt: str):
        super().__init__(prompt, category="default-chat-openai")
        self.evaluation_template = PromptTemplate(
            template="""You are an expert prompt evaluator. Analyze this prompt thoroughly.

PROMPT TO EVALUATE:
{prompt}

Evaluate each dimension and provide specific, actionable feedback.
Score each dimension from 1-10 and explain your reasoning.

Return strictly in this JSON format:
{{
    "scores": {{
        "clarity": <1-10>,
        "specificity": <1-10>,
        "completeness": <1-10>,
        "structure": <1-10>,
        "task_focus": <1-10>
    }},
    "feedback": {{
        "clarity": ["<specific feedback>", "<improvement needed>"],
        "specificity": ["<specific feedback>", "<improvement needed>"],
        "completeness": ["<specific feedback>", "<improvement needed>"],
        "structure": ["<specific feedback>", "<improvement needed>"],
        "task_focus": ["<specific feedback>", "<improvement needed>"]
    }},
    "improvement_suggestions": [
        "<actionable suggestion 1>",
        "<actionable suggestion 2>",
        "<actionable suggestion 3>"
    ]
}}""",
            input_variables=["prompt"]
        )

    def _create_progress_bar(self, score: float, max_length: int = 20) -> str:
        """Create a visual progress bar based on score"""
        filled = int(score * max_length / 10)
        return "█" * filled + "░" * (max_length - filled)

    def process(self, state: GraphState) -> GraphState:
        prompt = state.get_value("enhanced_prompt", self.prompt)
        
        print("\n" + "="*70)
        print(colored("🔍 PROMPT EVALUATION IN PROGRESS", "cyan", attrs=["bold"]))
        print("="*70)
        
        print(colored("\n📝 Original Prompt:", "yellow"))
        print(f"{prompt}\n")
        
        try:
            # Get evaluation from LLM
            evaluation_json = self.call_chat_openai(
                self.evaluation_template,
                {"prompt": prompt}
            )
            results = json.loads(evaluation_json)
            
            # Calculate overall score
            weights = {
                "clarity": 0.25,
                "specificity": 0.2,
                "completeness": 0.2,
                "structure": 0.15,
                "task_focus": 0.2
            }
            overall_score = sum(
                results["scores"][metric] * weight 
                for metric, weight in weights.items()
            )
            
            # Display Results
            print(colored("\n📊 EVALUATION SCORES", "cyan", attrs=["bold"]))
            print("-" * 40)
            
            # Overall Score
            score_color = "green" if overall_score >= 8 else "yellow" if overall_score >= 6 else "red"
            print(colored(f"\n🎯 Overall Score: {overall_score:.1f}/10", score_color, attrs=["bold"]))
            print(colored(self._create_progress_bar(overall_score, 30), score_color))
            
            # Individual Scores
            print(colored("\n📈 Dimension Scores:", "cyan"))
            for dimension, score in results["scores"].items():
                color = "green" if score >= 8 else "yellow" if score >= 6 else "red"
                print(f"{dimension.title():12} {colored(f'{score}/10 {self._create_progress_bar(score)}', color)}")
            
            # Detailed Feedback
            print(colored("\n💡 DETAILED FEEDBACK", "cyan", attrs=["bold"]))
            print("-" * 40)
            for dimension, feedback_list in results["feedback"].items():
                print(colored(f"\n{dimension.title()}:", "yellow"))
                for feedback in feedback_list:
                    print(f"  ✓ {feedback}")
            
            # Improvement Suggestions
            print(colored("\n🚀 IMPROVEMENT SUGGESTIONS", "cyan", attrs=["bold"]))
            print("-" * 40)
            for i, suggestion in enumerate(results["improvement_suggestions"], 1):
                print(f"{i}. {suggestion}")
            
            # Final Assessment
            print(colored("\n📋 FINAL ASSESSMENT", "cyan", attrs=["bold"]))
            print("-" * 40)
            if overall_score >= 8:
                print(colored("✨ Excellent prompt! Ready for use.", "green"))
            elif overall_score >= 6:
                print(colored("⚠️ Good prompt with room for improvement.", "yellow"))
            else:
                print(colored("❌ Significant improvements needed.", "red"))
            
            print("\n" + "="*70 + "\n")
            
            # Update state
            state.update_keys({
                "prompt_evaluation": {
                    "scores": results["scores"],
                    "overall_score": overall_score,
                    "feedback": results["feedback"],
                    "suggestions": results["improvement_suggestions"],
                    "needs_improvement": overall_score < 7.0
                }
            })
            
        except Exception as e:
            print(colored(f"Error in evaluation: {e}", "red"))
            # Set default evaluation results in case of error
            state.update_keys({
                "prompt_evaluation": {
                    "scores": {},
                    "overall_score": 0,
                    "feedback": {},
                    "suggestions": [],
                    "needs_improvement": True
                }
            })
        
        return state