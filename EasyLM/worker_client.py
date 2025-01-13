import os
infer_disk = os.environ.get("INFER_DISK")
os.environ["JAX_COMPILATION_CACHE_DIR"] = f"/mnt/{infer_disk}/jax_cache"

import logging
import asyncio
from typing import Dict, Optional, Set
import time
from concurrent.futures import ThreadPoolExecutor
import threading

import jax
import jax.numpy as jnp
import mlxu
from transformers import AutoTokenizer
import socketio

jax.config.update("jax_compilation_cache_dir", "/mnt/disk3/jax_cache")

from EasyLM.models.llama.llama_config import create_llama_flags
create_llama_flags()

from EasyLM.serving import LMServer

class WorkerClient:
    """Client that connects to coordinator and handles inference requests asynchronously."""
    
    def __init__(self, coordinator_url: str, model_server: LMServer):
        self.sio = socketio.AsyncClient(
            reconnection=True,
            reconnection_attempts=10,
            reconnection_delay=1,
            reconnection_delay_max=5
        )
        self.coordinator_url = coordinator_url
        self.model_server = model_server
        self.active_requests: Set[str] = set()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.lock = threading.Lock()
        self.heartbeat_task = None
        
        # Set up socketio event handlers
        @self.sio.event
        async def connect():
            logging.info(f"Connected to coordinator at {coordinator_url}")
            # Send LoRA checkpoint info
            await self.sio.emit('worker_info', {
                'lora_path': FLAGS.load_lora if FLAGS.lora_mode else 'No LoRA loaded'
            })
            # Start heartbeat when connected
            if self.heartbeat_task is None:
                self.heartbeat_task = asyncio.create_task(self._heartbeat())
            
        @self.sio.event
        async def disconnect():
            logging.info("Disconnected from coordinator")
            # Stop heartbeat on disconnect
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
                self.heartbeat_task = None
            
        @self.sio.event
        async def inference_request(data):
            request_id = data['request_id']
            if request_id in self.active_requests:
                logging.warning(f"Duplicate request {request_id}, ignoring")
                return
                
            logging.info(f"Received inference request: {request_id}")
            self.active_requests.add(request_id)
            
            # Create background task for inference
            asyncio.create_task(self._handle_inference(data))

    async def _handle_inference(self, data: Dict):
        """Handle inference request in background task."""
        request_id = data['request_id']
        try:
            # Run inference in thread pool without timeout
            response, _ = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.model_server.process_chat,
                data['prompt'],
                data['context'],
                data.get('temperature', None)
            )
            
            # Send response if still connected
            if self.sio.connected:
                await self.sio.emit('inference_response', {
                    'request_id': request_id,
                    'response': response,
                    'status': 'success'
                })
            else:
                logging.error(f"Cannot send response for {request_id} - disconnected")
                
        except asyncio.TimeoutError:
            logging.error(f"Inference timeout for request {request_id}")
            if self.sio.connected:
                await self.sio.emit('inference_response', {
                    'request_id': request_id,
                    'error': "Inference timeout - request took too long",
                    'status': 'timeout'
                })
                
        except Exception as e:
            logging.error(f"Error processing request {request_id}: {str(e)}")
            if self.sio.connected:
                await self.sio.emit('inference_response', {
                    'request_id': request_id,
                    'error': str(e),
                    'status': 'error'
                })
        finally:
            self.active_requests.remove(request_id)

    async def _heartbeat(self):
        """Send periodic heartbeats to keep connection alive."""
        while True:
            try:
                if self.sio.connected:
                    await self.sio.emit('heartbeat', {
                        'active_requests': list(self.active_requests),
                        'timestamp': time.time()
                    })
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in heartbeat: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying
            
    async def run(self):
        """Connect to coordinator and handle requests."""
        while True:
            try:
                await self.sio.connect(self.coordinator_url)
                await self.sio.wait()
            except Exception as e:
                logging.error(f"Connection error: {str(e)}")
                await asyncio.sleep(5)  # Wait before reconnecting
        
    def start(self):
        """Start the client in the current thread."""
        asyncio.run(self.run())

if __name__ == '__main__':
    import mlxu
    from EasyLM.models.llama.llama_serve import ModelServer, FLAGS
    from EasyLM.models.llama.llama_config import create_llama_flags
    
    def main(argv):
        logging.basicConfig(level=logging.INFO)
        server = ModelServer()
        client = WorkerClient(FLAGS.coordinator_url, server)
        client.start()

    mlxu.run(main)
