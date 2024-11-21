# app.py
from flask import Flask, request
from dotenv import load_dotenv
from time import time
import os
from flask_cors import CORS
from agentic.enhancer import AgenticEnhancer

# Load environment variables
load_dotenv()
DEBUG_MODE = os.getenv("DEBUG_MODE") == "true"
PORT = os.getenv("PORT", 5000)

app = Flask(__name__)

# Configure CORS
CORS(app, origins="*")


@app.route('/agentic_enhancer', methods=['POST'])
def agentic_enhancer():
    param = request.get_json()
    enhance = param.get('enhance', False)
    user_type = param.get('userType', "qfg-employee")
    chat_id = param.get('chatId', None)
    original_query = param.get('query', None)

    agentic = AgenticEnhancer(original_query, user_type, chat_id)
    agentic_answer = agentic.execute_workflow()

    answer = agentic_answer.get("answer")
    enhancedConfig = agentic_answer.get("enhancedConfig")
    enhanced_query = agentic_answer.get("enhanced_query")

    return {
        "answer": answer,
        "enhanced": enhance,
        "enhancedConfig": enhancedConfig,
        "enhanced_query": enhanced_query,
        "original_query": original_query
    }


if __name__ == '__main__':
    app.run(debug=DEBUG_MODE, host="0.0.0.0", port=PORT)
    print("Prompt enhancer API is running")
