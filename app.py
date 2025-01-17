from dotenv import load_dotenv
import os
from flask import Flask, request
from flask_cors import CORS

from agentic.enhancer import AgenticEnhancer

load_dotenv()

DEBUG_MODE = os.getenv("DEBUG_MODE") == "true"
PORT = os.getenv("PORT", 5000)

app = Flask(__name__)

# TODO - Add better cors handling
CORS(app, origins="*")


@app.route('/enhancer', methods=['POST'])
def agentic_enhancer():
    param = request.get_json()
    # chat_id = param.get('chatId', None)
    original_prompt = param.get('prompt', None)

    # agentic = AgenticEnhancer(original_prompt, chat_id)
    agentic = AgenticEnhancer(original_prompt)
    agentic_answer = agentic.execute_workflow()

    enhanced_prompt = agentic_answer.get("enhanced_prompt")
    # enhanced_config = agentic_answer.get("enhanced_config")
    answer = agentic_answer.get("final_prompt_answer")

    return {
        "originalPrompt": original_prompt,
        "enhancedPrompt": enhanced_prompt,
        # "enhancedConfig": enhanced_config,
        "answer": answer,
    }


@app.route('/health', methods=['GET'])
def healthcheck():
    return "healthy"


if __name__ == '__main__':
    app.run(debug=DEBUG_MODE, host="0.0.0.0", port=PORT)
    # logger.info("Prompt enhancer API is running")
    print("Prompt enhancer API is running")
