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
            
            # Do warmup generations after first worker connects
            if not self.warmup_done and len(self.connected_workers) > 0:
                logging.info("Performing warmup generations...")
                warmup_text = "<msg username=\"user\">Hello</msg>\n<msg username=\"simulect\">"
                for _ in range(2):  # Two warmup runs
                    try:
                        response = await self.sio.emit('inference_request', {
                            'request_id': -1,  # Special request ID for warmup
                            'prompt': warmup_text,
                            'context': '',
                            'temperature': None
                        })
                        logging.info("Warmup generation successful")
                    except Exception as e:
                        logging.error(f"Warmup generation failed: {str(e)}")
                self.warmup_done = True
            
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
        with gr.Blocks(analytics_enabled=False, title='Multi-User Chat Simulation') as chat_ui:
            # State for maintaining chat history
            state = gr.State([])
            
            with gr.Row():
                chatbot = gr.Chatbot(label='Chat history', height=500)
                with open('simulects.json', 'r') as f:
                    simulects = json.load(f)
                with gr.Column():
                    simulated_user = gr.Dropdown(
                        choices=simulects,
                        value='simulect',
                        label='Simulating User'
                    )
                    channel_history = gr.Textbox(
                        placeholder='Paste previous channel history here (XML format)...',
                        label='Channel History',
                        lines=5,
                        value=''
                    )
                    msg = gr.Textbox(
                        placeholder='Type your message here...',
                        label='Your message (as another user)',
                        show_label=True
                    )
                    username = gr.Textbox(
                        placeholder='Your username',
                        label='Your username',
                        value='user'
                    )
                    with gr.Row():
                        send = gr.Button('Send')
                        generate = gr.Button('Generate Next')
                        undo = gr.Button('Undo Last')
                        clear = gr.Button('Clear')
            
            def format_message(username, text):
                return f'<msg username="{username}">{text}</msg>'
            
            def format_history(history):
                return "\n".join(msg for msg in history)
            
            def user(user_message, username, history, simulated_user, channel_history):
                if not user_message.strip():
                    return "", history, history
                
                # Format and add user message
                formatted_msg = format_message(username, user_message)
                new_history = history + [formatted_msg]
                
                # Prepare prompt with channel history, current conversation, and opening tag
                history_text = format_history(new_history)
                opening_tag = f'<msg username="{simulated_user}">'
                
                if channel_history.strip():
                    prompt = channel_history.strip() + "\n" + history_text + "\n" + opening_tag
                else:
                    prompt = history_text + "\n" + opening_tag
                
                try:
                    # Get model response
                    response = requests.post(
                        f"http://localhost:{self.port}/chat",
                        json={"prompt": prompt}
                    ).json()
                    
                    logging.info("Response:")
                    logging.info(json.dumps(response, indent=2))
                    
                    # Extract just the response text, removing any XML tags
                    if isinstance(response, str):
                        # If response contains a complete message tag for wrong user, discard it
                        if f'username="' in response and f'username="{simulated_user}"' not in response:
                            response = ""
                        else:
                            # Clean up response if needed
                            response = response.replace('</msg>', '').strip()
                    
                    # Format model's response - add opening tag to prompt but not to history
                    if response:
                        # Add just the response text to history with full tags
                        formatted_response = format_message(simulated_user, response)
                        new_history.append(formatted_response)
                    
                    # Update chatbot display
                    chat_display = []
                    for msg in new_history:
                        if 'username="' in msg and '">' in msg and '</msg>' in msg:
                            username = msg.split('username="')[1].split('"')[0]
                            content = msg.split('">')[1].split('</msg>')[0]
                            chat_display.append([f"{username}: {content}", None])
                    
                    return "", chat_display, new_history
                    
                except Exception as e:
                    logging.error(f"Error in chat: {str(e)}")
                    return "", history, history
            
            msg.submit(
                user,
                [msg, username, state, simulated_user, channel_history],
                [msg, chatbot, state]
            )
            
            send.click(
                user,
                [msg, username, state, simulated_user, channel_history],
                [msg, chatbot, state]
            )
            
            def undo_last(history, chat_history):
                if len(history) >= 2:  # Remove last user message and model response
                    history = history[:-2]
                    chat_history = chat_history[:-2]
                return history, chat_history

            clear.click(
                lambda: ([], []),
                None,
                [state, chatbot]
            )
            
            def generate_next(history, simulated_user, channel_history):
                if not history:
                    return "", history, history
                    
                # Prepare prompt with channel history, current conversation, and opening tag
                history_text = format_history(history)
                opening_tag = f'<msg username="{simulated_user}">'
                
                if channel_history.strip():
                    prompt = channel_history.strip() + "\n" + history_text + "\n" + opening_tag
                else:
                    prompt = history_text + "\n" + opening_tag
                
                try:
                    # Get model response
                    response = requests.post(
                        f"http://localhost:{self.port}/chat",
                        json={"prompt": prompt}
                    ).json()
                    
                    logging.info("Response:")
                    logging.info(json.dumps(response, indent=2))
                    
                    # Extract just the response text, removing any XML tags
                    if isinstance(response, str):
                        # If response contains a complete message tag for wrong user, discard it
                        if f'username="' in response and f'username="{simulated_user}"' not in response:
                            response = ""
                        else:
                            # Clean up response if needed
                            # Remove any complete message tags
                            if '<msg' in response and '</msg>' in response:
                                response = response.split('">')[1].split('</msg>')[0]
                    
                    # Format model's response - add opening tag to prompt but not to history
                    if response:
                        # Add just the response text to history with full tags
                        formatted_response = format_message(simulated_user, response)
                        new_history = history + [formatted_response]
                    else:
                        new_history = history
                    
                    # Update chatbot display
                    chat_display = []
                    for msg in new_history:
                        if 'username="' in msg and '">' in msg and '</msg>' in msg:
                            username = msg.split('username="')[1].split('"')[0]
                            content = msg.split('">')[1].split('</msg>')[0]
                            chat_display.append([f"{username}: {content}", None])
                    
                    return "", chat_display, new_history
                    
                except Exception as e:
                    logging.error(f"Error in chat: {str(e)}")
                    return "", history, history

            undo.click(
                undo_last,
                [state, chatbot],
                [state, chatbot]
            )
            
            generate.click(
                generate_next,
                [state, simulated_user, channel_history],
                [msg, chatbot, state]
            )
            
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
