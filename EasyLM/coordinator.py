import logging
from typing import Dict, List, Optional
import asyncio
import json
import requests

import socketio
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel

class InferenceRequest(BaseModel):
    prompt: str
    context: str = ''
    temperature: Optional[float] = None

class CoordinatorServer:
    """Central server that coordinates distributed inference across workers."""
    
    def __init__(self, host='0.0.0.0', port=5010):
        self.host = host
        self.port = port
        self.sio = socketio.AsyncServer(async_mode='asgi')
        self.app = FastAPI()
        self.sio_app = socketio.ASGIApp(self.sio, self.app)
        self.connected_workers = set()
        self.active_requests = {}
        self.request_counter = 0
        
        # Set up socketio event handlers
        @self.sio.event
        async def connect(sid, environ):
            logging.info(f"Worker {sid} connected")
            self.connected_workers.add(sid)
            
        @self.sio.event
        async def disconnect(sid):
            logging.info(f"Worker {sid} disconnected")
            self.connected_workers.remove(sid)
            
        @self.sio.event
        async def inference_response(sid, data):
            request_id = data['request_id']
            if request_id in self.active_requests:
                self.active_requests[request_id]['responses'][sid] = data['response']
                logging.info(f"Received response from worker {sid}")
                logging.info(json.dumps(data, indent=2))
                # Check if we have responses from all workers
                if len(self.active_requests[request_id]['responses']) == len(self.connected_workers):
                    # All workers responded, resolve the future
                    self.active_requests[request_id]['future'].set_result(
                        self.active_requests[request_id]['responses']
                    )
                    
        # Set up HTTP endpoints
        @self.app.post("/chat")
        async def chat(request: InferenceRequest):
            if not self.connected_workers:
                return {"error": "No workers connected"}
                
            # Create new request
            request_id = self.request_counter
            self.request_counter += 1
            future = asyncio.Future()
            self.active_requests[request_id] = {
                'future': future,
                'responses': {}
            }
            
            # Broadcast request to all workers
            await self.sio.emit('inference_request', {
                'request_id': request_id,
                'prompt': request.prompt,
                'context': request.context,
                'temperature': request.temperature
            })
            
            # Wait for responses from all workers
            try:
                responses = await asyncio.wait_for(future, timeout=30.0)
                # All responses should be identical since workers run same code
                # Just return the first one
                return responses[next(iter(responses))]
            except asyncio.TimeoutError:
                del self.active_requests[request_id]
                return {"error": "Request timed out"}
                
        # Add Gradio chat interface
        self.app = gr.mount_gradio_app(
            self.app,
            self.create_chat_interface(),
            path="/chat-ui"
        )
        
    def create_chat_interface(self):
        with gr.Blocks(analytics_enabled=False, title='Distributed LLM Chat') as chat_ui:
            chatbot = gr.Chatbot(label='Chat history')
            msg = gr.Textbox(placeholder='Type your message here...', show_label=False)
            with gr.Row():
                send = gr.Button('Send')
                clear = gr.Button('Clear')
            
            def user(user_message, history):
                return "", history + [[user_message, None]]
                
            def bot(history):
                user_message = history[-1][0]
                try:
                    response = requests.post(
                        f"http://localhost:{self.port}/chat",
                        json={"prompt": user_message}
                    ).json()
                    logging.info("Response:")
                    logging.info(json.dumps(response, indent=2))
                    # if 'error' in response:
                    #     history[-1][1] = f"Error: {response['error']}"
                    # else:
                    history[-1][1] = response
                except Exception as e:
                    history[-1][1] = f"Error: {str(e)}"
                return history
            
            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, chatbot, chatbot
            )
            send.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, chatbot, chatbot
            )
            clear.click(lambda: None, None, chatbot, queue=False)
            
        return chat_ui
        
    def run(self):
        """Start the coordinator server."""
        import uvicorn
        uvicorn.run(
            self.sio_app,
            host=self.host,
            port=self.port
        )

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    server = CoordinatorServer()
    server.run()
