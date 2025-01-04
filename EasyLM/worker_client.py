import logging
from typing import Dict, Optional
import asyncio
import json

import socketio
import jax
import jax.numpy as jnp
import mlxu

from EasyLM.serving import LMServer

class WorkerClient:
    """Client that connects to coordinator and handles inference requests."""
    
    def __init__(self, coordinator_url: str, model_server: LMServer):
        self.sio = socketio.AsyncClient()
        self.coordinator_url = coordinator_url
        self.model_server = model_server
        
        # Set up socketio event handlers
        @self.sio.event
        async def connect():
            logging.info(f"Connected to coordinator at {coordinator_url}")
            
        @self.sio.event
        async def disconnect():
            logging.info("Disconnected from coordinator")
            
        @self.sio.event
        async def inference_request(data):
            logging.info(f"Received inference request: {data}")
            request_id = data['request_id']
            
            # Process request using model server
            response, _ = self.model_server.process_chat(
                data['prompt'],
                data['context'],
                data.get('temperature', None)
            )
            
            # Send response back to coordinator
            await self.sio.emit('inference_response', {
                'request_id': request_id,
                'response': response
            })
            
    async def run(self):
        """Connect to coordinator and handle requests."""
        await self.sio.connect(self.coordinator_url)
        await self.sio.wait()
        
    def start(self):
        """Start the client in the current thread."""
        asyncio.run(self.run())

if __name__ == '__main__':
    import mlxu
    from EasyLM.models.llama.llama_serve import ModelServer, FLAGS, FLAGS_DEF
    
    # Add coordinator URL to flags
    FLAGS_DEF.coordinator_url = 'http://localhost:5010'
    
    def main(argv):
        logging.basicConfig(level=logging.INFO)
        
        server = ModelServer(FLAGS.lm_server)
        
        # Start worker client
        client = WorkerClient(FLAGS.coordinator_url, server)
        client.start()

    mlxu.run(main)
