import logging
from typing import Dict, Optional
import asyncio
import json

import socketio
import jax
import jax.numpy as jnp

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--coordinator-url', default='http://localhost:5010')
    parser.add_argument('--mesh_dim', default='1,-1,1')
    parser.add_argument('--dtype', default='bf16')
    parser.add_argument('--llama.base_model', default='llama32_1b')
    parser.add_argument('--tokenizer')
    parser.add_argument('--load_checkpoint')
    parser.add_argument('--input_length', type=int, default=1024)
    parser.add_argument('--seq_length', type=int, default=2048)
    parser.add_argument('--do_sample', type=bool, default=True)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.95)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Import and initialize flags/model server
    from EasyLM.models.llama.llama_serve import ModelServer
    from EasyLM.models.llama.llama_config import create_llama_flags
    
    # Create flags with command line args as updates
    updates = {
        'mesh_dim': args.mesh_dim,
        'dtype': args.dtype,
        'llama': {'base_model': getattr(args, 'llama.base_model')},
        'tokenizer': args.tokenizer,
        'load_checkpoint': args.load_checkpoint,
        'input_length': args.input_length,
        'seq_length': args.seq_length,
        'do_sample': args.do_sample,
        'top_k': args.top_k,
        'top_p': args.top_p,
    }
    FLAGS, _ = create_llama_flags(updates)
    
    server = ModelServer(FLAGS.lm_server)
    
    # Start worker client
    client = WorkerClient(args.coordinator_url, server)
    client.start()
