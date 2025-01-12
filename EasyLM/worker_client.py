import os
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/mnt/disk2/jax_cache"
#os.environ["JAX_DEBUG_LOG_MODULES"] = "jax._src.compiler,jax._src.lru_cache"

import logging
import jax
import jax.numpy as jnp
import mlxu
from transformers import AutoTokenizer
import socketio

jax.config.update("jax_compilation_cache_dir", "/mnt/disk2/jax_cache")
#jax.config.update("jax_explain_cache_misses", True)
#jax.config.update("jax_persistent_cache_enable_xla_caches", "all")


from EasyLM.models.llama.llama_config import create_llama_flags
create_llama_flags()

from typing import Dict, Optional
import asyncio

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
    from EasyLM.models.llama.llama_serve import ModelServer, FLAGS
    from EasyLM.models.llama.llama_config import create_llama_flags
    
    # Import existing flags and add coordinator URL
    from EasyLM.models.llama.llama_serve import FLAGS
    
    def main(argv):
        logging.basicConfig(level=logging.INFO)
        
        server = ModelServer()
        
        # Start worker client
        client = WorkerClient(FLAGS.coordinator_url, server)
        client.start()

    mlxu.run(main)
