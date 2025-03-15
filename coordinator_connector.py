import os
import json
import time
import asyncio
import logging
import socketio
import aiohttp
import uuid
from typing import Dict, Any, Optional, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoordinatorConnector:
    """
    Connects the tree conversation backend with the EasyLM coordinator service.
    Acts as a client to the coordinator and provides an interface for text generation.
    """
    
    def __init__(self, coordinator_url: str = "http://localhost:5010"):
        self.coordinator_url = coordinator_url
        self.session = None
        self.active_requests = {}
        self.request_lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the connector with an HTTP session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        
        # Check if coordinator is available
        try:
            async with self.session.get(f"{self.coordinator_url}/ready", timeout=5) as response:
                if response.status == 200:
                    logger.info(f"Successfully connected to coordinator at {self.coordinator_url}")
                    return True
        except Exception as e:
            logger.error(f"Failed to connect to coordinator: {str(e)}")
            return False
    
    async def generate_text(self, 
                          prompt: str, 
                          context: str = "",
                          temperature: Optional[float] = None,
                          top_p: Optional[float] = 0.95,
                          top_k: Optional[int] = 50,
                          max_tokens: Optional[int] = 1024,
                          stop_sequences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Send a generation request to the coordinator service.
        Returns the generated text or an error message.
        """
        if self.session is None:
            await self.initialize()
        
        request_id = str(uuid.uuid4())
        
        try:
            # Create the request payload
            payload = {
                "prompt": prompt,
                "context": context,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_tokens": max_tokens
            }
            
            # Add stop sequences if provided
            if stop_sequences:
                payload["stop_sequences"] = stop_sequences
            
            # Send the request to the coordinator
            async with self.session.post(
                f"{self.coordinator_url}/chat",
                json=payload,
                timeout=60  # 1 minute timeout
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if isinstance(result, dict) and "error" in result:
                        logger.error(f"Generation error: {result['error']}")
                        return {"status": "error", "message": result["error"]}
                    return {"status": "success", "text": result}
                else:
                    error_text = await response.text()
                    logger.error(f"Generation failed with status {response.status}: {error_text}")
                    return {"status": "error", "message": f"HTTP error {response.status}: {error_text}"}
        
        except asyncio.TimeoutError:
            logger.error(f"Generation request timed out")
            return {"status": "error", "message": "Request timed out"}
        
        except Exception as e:
            logger.error(f"Error in generate_text: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def get_worker_status(self) -> Dict[str, Any]:
        """Get the status of all connected workers."""
        if self.session is None:
            await self.initialize()
        
        try:
            async with self.session.get(f"{self.coordinator_url}/workers", timeout=5) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"status": "success", "workers": result}
                else:
                    error_text = await response.text()
                    logger.error(f"Get worker status failed with status {response.status}: {error_text}")
                    return {"status": "error", "message": f"HTTP error {response.status}: {error_text}"}
        
        except Exception as e:
            logger.error(f"Error in get_worker_status: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def close(self):
        """Close the connector and any open connections."""
        if self.session:
            await self.session.close()
            self.session = None

# Integration with the tree conversation backend
async def integrate_with_tree_backend(app, data_store):
    """
    Integrate the coordinator connector with the tree conversation backend.
    This replaces the simulated generation in the request_generation Socket.IO event.
    """
    # Create and initialize the connector with specified coordinator URL
    coordinator_url = os.environ.get("COORDINATOR_URL", "http://51.81.181.136:5010")
    logger.info(f"Connecting to coordinator at {coordinator_url}")
    connector = CoordinatorConnector(coordinator_url=coordinator_url)
    await connector.initialize()
    
    # Store the connector in the app state
    app.state.coordinator_connector = connector
    
    # Get socketio instance directly
    from tree_conversation_backend import sio
    
    # Modify the request_generation event handler
    @sio.event
    async def request_generation(sid, data):
        """Request text generation from the coordinator."""
        user_id = data.get('user_id')
        conversation_id = data.get('conversation_id')
        parent_id = data.get('parent_id')
        prompt = data.get('prompt')
        settings = data.get('settings', {})
        
        if not all([user_id, conversation_id, parent_id, prompt]):
            return {"status": "error", "message": "Missing required fields"}
        
        # Create a user message node first
        user_message = Message(
            content=prompt,
            role="user",
            metadata=settings
        )
        
        user_node = data_store.add_node(
            conversation_id=conversation_id,
            parent_id=parent_id,
            message=user_message,
            user_id=user_id
        )
        
        if not user_node:
            return {"status": "error", "message": "Failed to add user message"}
        
        # Broadcast to everyone in the conversation
        await sio.emit('node:create', {
            "conversation_id": conversation_id,
            "node": user_node.dict()
        }, room=f"conversation:{conversation_id}")
        
        # Notify clients that generation is starting
        await sio.emit('generation:start', {
            "conversation_id": conversation_id,
            "parent_id": user_node.id
        }, room=f"conversation:{conversation_id}")
        
        try:
            # Get the full conversation history up to this point
            branch = data_store.get_branch(conversation_id, user_node.id)
            context = "\n".join([node.message.content for node in branch[:-1]])  # Exclude the user's message
            
            # Extract generation parameters from settings
            temperature = settings.get('temperature', 0.8)
            top_p = settings.get('top_p', 0.95)
            top_k = settings.get('top_k', 50)
            max_tokens = settings.get('max_tokens', 1024)
            
            # Send the request to the coordinator service
            result = await app.state.coordinator_connector.generate_text(
                prompt=prompt,
                context=context,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens
            )
            
            if result["status"] == "error":
                # Handle generation error
                await sio.emit('generation:error', {
                    "conversation_id": conversation_id,
                    "parent_id": user_node.id,
                    "error": result["message"]
                }, room=f"conversation:{conversation_id}")
                return {"status": "error", "message": result["message"]}
            
            # Create assistant message with the generated text
            assistant_message = Message(
                content=result["text"],
                role="assistant",
                metadata=settings
            )
            
            assistant_node = data_store.add_node(
                conversation_id=conversation_id,
                parent_id=user_node.id,
                message=assistant_message,
                user_id="system"
            )
            
            # Broadcast to everyone in the conversation
            await sio.emit('node:create', {
                "conversation_id": conversation_id,
                "node": assistant_node.dict()
            }, room=f"conversation:{conversation_id}")
            
            # Notify clients that generation is complete
            await sio.emit('generation:complete', {
                "conversation_id": conversation_id,
                "node_id": assistant_node.id
            }, room=f"conversation:{conversation_id}")
            
            return {"status": "success", "node": assistant_node.dict()}
        
        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            await sio.emit('generation:error', {
                "conversation_id": conversation_id,
                "parent_id": user_node.id,
                "error": str(e)
            }, room=f"conversation:{conversation_id}")
            
            return {"status": "error", "message": str(e)}
    
    # Add worker status endpoint
    @app.get("/api/workers")
    async def get_worker_status():
        """Get status of all coordinator workers."""
        result = await app.state.coordinator_connector.get_worker_status()
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        return result["workers"]
    
    # Add cleanup on shutdown
    @app.on_event("shutdown")
    async def shutdown_event():
        await app.state.coordinator_connector.close()
    
    return app

# If this script is run directly, it will start the integrated service
if __name__ == "__main__":
    import uvicorn
    from tree_conversation_backend import app, application, data_store, Message
    from fastapi import HTTPException
    import asyncio
    
    # Set up the integration
    loop = asyncio.get_event_loop()
    loop.run_until_complete(integrate_with_tree_backend(app, data_store))
    
    # Start the server without specifying the loop
    uvicorn.run(application, host="0.0.0.0", port=8000)