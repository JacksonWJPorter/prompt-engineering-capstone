from dotenv import load_dotenv
import os
import time
import json
from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from agentic.enhancer import AgenticEnhancer

load_dotenv()

DEBUG_MODE = os.getenv("DEBUG_MODE") == "true"
PORT = int(os.getenv("PORT", 5000))

app = Flask(__name__)
# Enable CORS for all routes and origins
CORS(app, origins="*", supports_credentials=True)
# Initialize SocketIO with CORS support and debug logging
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=True,
    ping_timeout=60,
    ping_interval=25,
    async_mode='threading'  # Use threading mode for better compatibility
)

# Store active connections for debugging
active_connections = {}

# Custom event emitter class for node tracking


class NodeEventEmitter:
    def __init__(self, socket_io_instance, session_id="unknown"):
        self.socketio = socket_io_instance
        self.session_id = session_id
        print(f"Created NodeEventEmitter for session {session_id}")

    def emit_node_completion(self, node_name, node_data):
        """Emit a node completion event to connected clients"""
        print(f"[EMIT] Sending node_completed event for {node_name}")
        try:
            # Ensure node_data is serializable
            if node_data:
                # Try to convert to dict if not already
                if not isinstance(node_data, dict):
                    try:
                        node_data = json.loads(json.dumps(node_data))
                    except:
                        node_data = {"data": str(node_data)}

            # Add additional metadata
            event_data = {
                'node_name': node_name,
                'node_type': node_data.get('node_type', node_name),
                'node_output': node_data.get('node_output', None),
                'timestamp': time.time(),
                'session_id': self.session_id
            }

            # Log the actual data being sent
            print(f"[EMIT] Data payload: {json.dumps(event_data)[:1000]}...")

            # Emit the event
            self.socketio.emit('node_completed', event_data)
            print(f"[EMIT] Successfully emitted event for {node_name}")

            # Send a confirmation event (for debugging)
            self.socketio.emit('emit_confirmation', {
                'message': f'Emitted {node_name} event',
                'timestamp': time.time()
            })
        except Exception as e:
            print(f"[EMIT ERROR] Failed to emit event: {str(e)}")


@app.route('/enhancer', methods=['POST'])
def agentic_enhancer():
    param = request.get_json()
    original_prompt = param.get('prompt', None)
    session_id = param.get('sessionId', 'default-session')

    print(f"Received request with session ID: {session_id}")

    # Log active connections
    print(f"Active connections: {list(active_connections.keys())}")

    # Create event emitter instance with session_id
    event_emitter = NodeEventEmitter(socketio, session_id)
    print(f"Created event emitter for session: {session_id}")

    # Initialize the enhancer with the emitter
    agentic = AgenticEnhancer(original_prompt, event_emitter=event_emitter)

    try:
        # Send periodic status updates during processing
        def send_status_update():
            socketio.emit('status_update', {
                'message': 'Processing prompt...',
                'timestamp': time.time(),
                'session_id': session_id
            })

        # Send initial status update
        send_status_update()

        # Execute the workflow
        agentic_answer = agentic.execute_workflow()

        enhanced_prompt = agentic_answer.get("enhanced_prompt")

        # Send completion event
        socketio.emit('processing_completed', {
            'message': 'Prompt enhancement completed',
            'session_id': session_id,
            'timestamp': time.time()
        })

        return {
            "originalPrompt": original_prompt,
            "enhancedPrompt": enhanced_prompt,
        }
    except Exception as e:
        print(f"Error executing workflow: {str(e)}")
        # Send error event
        socketio.emit('processing_error', {
            'message': f'Error: {str(e)}',
            'session_id': session_id,
            'timestamp': time.time()
        })
        # Return a more helpful error response
        return {
            "originalPrompt": original_prompt,
            "enhancedPrompt": original_prompt,
            "answer": f"Error: {str(e)}",
            "error": True
        }, 500

# Socket connection handlers


@socketio.on('connect')
def handle_connect():
    sid = request.sid
    active_connections[sid] = {
        'connected_at': time.time(),
        'ip': request.remote_addr if hasattr(request, 'remote_addr') else 'unknown'
    }
    print(f'Client connected with SID: {sid}')
    # Send welcome message
    emit('welcome', {
        'message': 'Connected to prompt enhancer server',
        'sid': sid,
        'timestamp': time.time()
    })


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in active_connections:
        del active_connections[sid]
    print(f'Client disconnected: {sid}')


@socketio.on('test_connection')
def handle_test_connection(data):
    print(f"Test connection received: {data}")
    # Echo back to confirm receipt
    emit('test_connection_response', {
        'message': 'Hello from prompt enhancer server',
        'received': data,
        'timestamp': time.time()
    })

# Add a ping endpoint for client heartbeat


@app.route('/ping', methods=['GET'])
def ping():
    return {'status': 'ok', 'time': time.time()}, 200

# Add an endpoint to get current server status


@app.route('/status', methods=['GET'])
def server_status():
    return {
        'status': 'running',
        'connections': len(active_connections),
        'time': time.time()
    }, 200


if __name__ == '__main__':
    # Print startup message
    print(f"Starting Prompt Enhancer API on port {PORT}")
    print(f"Debug mode: {DEBUG_MODE}")
    print(f"CORS origins: *")

    # Using socketio.run instead of app.run for WebSocket support
    socketio.run(app, debug=DEBUG_MODE, host="0.0.0.0", port=PORT)
    print("Prompt enhancer API is running with WebSocket support")
