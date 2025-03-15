import os
import json
import time
import uuid
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from pydantic import BaseModel, Field
import threading
import socketio
import uvicorn
from fastapi import FastAPI, HTTPException, Body, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data models
class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    role: str  # "user", "assistant", "system"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ConversationNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    message: Message
    created_by: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    branch_name: Optional[str] = None

class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    nodes: Dict[str, ConversationNode] = Field(default_factory=dict)
    root_id: Optional[str] = None
    created_by: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    settings: Dict[str, Any] = Field(default_factory=dict)

class GenerationRequest(BaseModel):
    conversation_id: str
    parent_node_id: str
    prompt: str
    settings: Dict[str, Any] = Field(default_factory=dict)

class UserPresence(BaseModel):
    user_id: str
    name: str
    active_conversation: Optional[str] = None
    last_active: str = Field(default_factory=lambda: datetime.now().isoformat())

# In-memory data store with JSON persistence
class DataStore:
    def __init__(self, data_dir: str = "conversation_data"):
        self.data_dir = data_dir
        self.conversations: Dict[str, Conversation] = {}
        self.users: Dict[str, UserPresence] = {}
        self.save_lock = threading.Lock()
        self.last_save_time = 0
        self.save_interval = 60  # seconds
        
        # Make sure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing conversations
        self._load_conversations()
    
    def _load_conversations(self):
        """Load all conversations from disk."""
        try:
            conv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
            for file in conv_files:
                try:
                    with open(os.path.join(self.data_dir, file), 'r') as f:
                        data = json.load(f)
                        conv = Conversation.parse_obj(data)
                        self.conversations[conv.id] = conv
                        logger.info(f"Loaded conversation: {conv.id} - {conv.title}")
                except Exception as e:
                    logger.error(f"Error loading conversation from {file}: {str(e)}")
            logger.info(f"Loaded {len(self.conversations)} conversations")
        except Exception as e:
            logger.error(f"Error loading conversations: {str(e)}")
    
    def _save_conversation(self, conversation_id: str):
        """Save a specific conversation to disk."""
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            return
        
        try:
            with self.save_lock:
                filepath = os.path.join(self.data_dir, f"{conversation_id}.json")
                with open(filepath, 'w') as f:
                    f.write(conversation.json(indent=2))
                self.last_save_time = time.time()
        except Exception as e:
            logger.error(f"Error saving conversation {conversation_id}: {str(e)}")
    
    def save_all_if_needed(self):
        """Save all conversations if save interval has passed."""
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            with self.save_lock:
                for conv_id in self.conversations:
                    self._save_conversation(conv_id)
                self.last_save_time = current_time
    
    # Conversation operations
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return self.conversations.get(conversation_id)
    
    def list_conversations(self) -> List[Conversation]:
        """List all conversations."""
        return list(self.conversations.values())
    
    def create_conversation(self, title: str, user_id: str, 
                            initial_message: Optional[str] = None,
                            settings: Optional[Dict[str, Any]] = None) -> Conversation:
        """Create a new conversation with optional initial message."""
        conv = Conversation(
            title=title,
            created_by=user_id,
            settings=settings or {}
        )
        
        # Add initial message if provided
        if initial_message:
            root_message = Message(
                content=initial_message,
                role="system"
            )
            root_node = ConversationNode(
                message=root_message,
                created_by=user_id
            )
            conv.nodes[root_node.id] = root_node
            conv.root_id = root_node.id
        
        self.conversations[conv.id] = conv
        self._save_conversation(conv.id)
        return conv
    
    def update_conversation(self, conversation_id: str, 
                           updates: Dict[str, Any]) -> Optional[Conversation]:
        """Update conversation details."""
        conv = self.conversations.get(conversation_id)
        if not conv:
            return None
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(conv, key) and key not in ['id', 'nodes', 'created_at']:
                setattr(conv, key, value)
        
        conv.updated_at = datetime.now().isoformat()
        self._save_conversation(conversation_id)
        return conv
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        if conversation_id not in self.conversations:
            return False
        
        # Remove from memory
        del self.conversations[conversation_id]
        
        # Remove from disk
        try:
            os.remove(os.path.join(self.data_dir, f"{conversation_id}.json"))
            return True
        except Exception as e:
            logger.error(f"Error deleting conversation file: {str(e)}")
            return False
    
    # Node operations
    def add_node(self, conversation_id: str, parent_id: str, 
                message: Message, user_id: str,
                branch_name: Optional[str] = None) -> Optional[ConversationNode]:
        """Add a new node to a conversation."""
        conv = self.conversations.get(conversation_id)
        if not conv:
            return None
        
        # Verify parent exists if specified
        if parent_id and parent_id not in conv.nodes:
            return None
        
        node = ConversationNode(
            parent_id=parent_id,
            message=message,
            created_by=user_id,
            branch_name=branch_name
        )
        
        # If this is the first node, set it as root
        if not conv.root_id:
            conv.root_id = node.id
        
        conv.nodes[node.id] = node
        conv.updated_at = datetime.now().isoformat()
        self._save_conversation(conversation_id)
        return node
    
    def get_node(self, conversation_id: str, node_id: str) -> Optional[ConversationNode]:
        """Get a specific node."""
        conv = self.conversations.get(conversation_id)
        if not conv:
            return None
        return conv.nodes.get(node_id)
    
    def delete_node(self, conversation_id: str, node_id: str) -> bool:
        """Delete a node and all its children."""
        conv = self.conversations.get(conversation_id)
        if not conv or node_id not in conv.nodes:
            return False
        
        # Find all children of this node
        to_delete = [node_id]
        self._find_children(conv, node_id, to_delete)
        
        # Delete the nodes
        for nid in to_delete:
            del conv.nodes[nid]
        
        # If we deleted the root, update root_id
        if node_id == conv.root_id:
            conv.root_id = None if not conv.nodes else next(iter(conv.nodes))
        
        conv.updated_at = datetime.now().isoformat()
        self._save_conversation(conversation_id)
        return True
    
    def _find_children(self, conv: Conversation, parent_id: str, result: List[str]):
        """Recursively find all children of a node."""
        for nid, node in conv.nodes.items():
            if node.parent_id == parent_id:
                result.append(nid)
                self._find_children(conv, nid, result)
    
    def get_branch(self, conversation_id: str, node_id: str) -> List[ConversationNode]:
        """Get all nodes in a branch, from root to the specified node."""
        conv = self.conversations.get(conversation_id)
        if not conv or node_id not in conv.nodes:
            return []
        
        # Start from the target node and follow parents
        branch = []
        current_id = node_id
        
        while current_id:
            node = conv.nodes.get(current_id)
            if not node:
                break
            
            branch.append(node)
            current_id = node.parent_id
        
        # Reverse to get root-to-leaf order
        branch.reverse()
        return branch
    
    # User operations
    def register_user(self, user_id: str, name: str) -> UserPresence:
        """Register a user's presence."""
        user = UserPresence(
            user_id=user_id,
            name=name
        )
        self.users[user_id] = user
        return user
    
    def update_user_presence(self, user_id: str, 
                             conversation_id: Optional[str] = None) -> Optional[UserPresence]:
        """Update a user's active conversation and last active time."""
        if user_id not in self.users:
            return None
        
        user = self.users[user_id]
        user.last_active = datetime.now().isoformat()
        
        if conversation_id is not None:
            user.active_conversation = conversation_id
        
        return user
    
    def get_active_users(self, conversation_id: Optional[str] = None) -> List[UserPresence]:
        """Get all active users, optionally filtered by conversation."""
        active_time = datetime.now().timestamp() - (5 * 60)  # 5 minutes
        
        active_users = []
        for user in self.users.values():
            user_time = datetime.fromisoformat(user.last_active).timestamp()
            if user_time >= active_time:
                if conversation_id is None or user.active_conversation == conversation_id:
                    active_users.append(user)
        
        return active_users

# Create FastAPI app and Socket.IO server
app = FastAPI(title="EasyLM Tree Conversation Backend")
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
socket_app = socketio.ASGIApp(sio, app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (including the UI)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add route for serving the UI
@app.get("/tree_conversation_ui.html")
async def get_ui():
    ui_path = os.path.join(current_dir, "tree_conversation_ui.html")
    return FileResponse(ui_path)

# Serve static files (if any)
os.makedirs(os.path.join(current_dir, "static"), exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static")

# Initialize data store
data_store = DataStore()

# Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    logger.info(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    logger.info(f"Client disconnected: {sid}")

@sio.event
async def register_user(sid, data):
    """Register a user's presence."""
    user_id = data.get('user_id')
    name = data.get('name')
    
    if not user_id or not name:
        return {"status": "error", "message": "Missing user_id or name"}
    
    user = data_store.register_user(user_id, name)
    await sio.emit('user:join', user.dict())
    return {"status": "success", "user": user.dict()}

@sio.event
async def join_conversation(sid, data):
    """Join a conversation."""
    user_id = data.get('user_id')
    conversation_id = data.get('conversation_id')
    
    if not user_id or not conversation_id:
        return {"status": "error", "message": "Missing user_id or conversation_id"}
    
    # Verify conversation exists
    conversation = data_store.get_conversation(conversation_id)
    if not conversation:
        return {"status": "error", "message": "Conversation not found"}
    
    # Update user presence
    user = data_store.update_user_presence(user_id, conversation_id)
    if not user:
        return {"status": "error", "message": "User not found"}
    
    # Get active users in this conversation
    active_users = data_store.get_active_users(conversation_id)
    
    # Join the socket.io room for this conversation
    sio.enter_room(sid, f"conversation:{conversation_id}")
    
    # Notify others in the room
    await sio.emit('user:join', user.dict(), room=f"conversation:{conversation_id}")
    
    return {
        "status": "success", 
        "conversation": conversation.dict(),
        "active_users": [u.dict() for u in active_users]
    }

@sio.event
async def leave_conversation(sid, data):
    """Leave a conversation."""
    user_id = data.get('user_id')
    conversation_id = data.get('conversation_id')
    
    if not user_id or not conversation_id:
        return {"status": "error", "message": "Missing user_id or conversation_id"}
    
    # Update user presence
    user = data_store.update_user_presence(user_id, None)
    if not user:
        return {"status": "error", "message": "User not found"}
    
    # Leave the socket.io room
    sio.leave_room(sid, f"conversation:{conversation_id}")
    
    # Notify others in the room
    await sio.emit('user:leave', user.dict(), room=f"conversation:{conversation_id}")
    
    return {"status": "success"}

@sio.event
async def send_message(sid, data):
    """Send a message (create a node)."""
    user_id = data.get('user_id')
    conversation_id = data.get('conversation_id')
    parent_id = data.get('parent_id')
    content = data.get('content')
    role = data.get('role', 'user')
    branch_name = data.get('branch_name')
    
    if not all([user_id, conversation_id, content]):
        return {"status": "error", "message": "Missing required fields"}
    
    # Create the message
    message = Message(
        content=content,
        role=role,
        metadata=data.get('metadata', {})
    )
    
    # Add node to conversation
    node = data_store.add_node(
        conversation_id=conversation_id,
        parent_id=parent_id,
        message=message,
        user_id=user_id,
        branch_name=branch_name
    )
    
    if not node:
        return {"status": "error", "message": "Failed to add message"}
    
    # Broadcast to everyone in the conversation
    await sio.emit('node:create', {
        "conversation_id": conversation_id,
        "node": node.dict()
    }, room=f"conversation:{conversation_id}")
    
    return {"status": "success", "node": node.dict()}

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
        # TODO: Integrate with your coordinator service here
        # For now, we'll just simulate a response after a delay
        await asyncio.sleep(1)
        
        # Create assistant message
        assistant_message = Message(
            content=f"This is a simulated response to: {prompt}",
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

# FastAPI endpoints
@app.get("/api/conversations")
async def list_conversations():
    """List all conversations."""
    return {"conversations": [conv.dict() for conv in data_store.list_conversations()]}

@app.post("/api/conversations")
async def create_conversation(
    title: str = Body(...),
    user_id: str = Body(...),
    initial_message: Optional[str] = Body(None),
    settings: Optional[Dict[str, Any]] = Body({})
):
    """Create a new conversation."""
    conversation = data_store.create_conversation(
        title=title,
        user_id=user_id,
        initial_message=initial_message,
        settings=settings
    )
    return {"conversation": conversation.dict()}

@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a conversation by ID."""
    conversation = data_store.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation": conversation.dict()}

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    success = data_store.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "success"}

@app.get("/api/conversations/{conversation_id}/branches")
async def get_branches(
    conversation_id: str,
    node_id: Optional[str] = Query(None)
):
    """Get branches from a conversation, optionally starting from a specific node."""
    conversation = data_store.get_conversation(conversation_id)
    
@app.post("/api/conversations/{conversation_id}/branches")
async def create_branch(
    conversation_id: str,
    parent_id: str = Body(...),
    user_id: str = Body(...),
    branch_name: str = Body(...)
):
    """Create a new branch in a conversation."""
    conversation = data_store.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Create a new message to start the branch
    message = Message(
        content=f"New branch created: {branch_name}",
        role="system"
    )
    
    # Add the node to the conversation
    node = data_store.add_node(
        conversation_id=conversation_id,
        parent_id=parent_id,
        message=message,
        user_id=user_id,
        branch_name=branch_name
    )
    
    if not node:
        raise HTTPException(status_code=404, detail="Failed to create branch")
    
    return {"node": node.dict()}
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if node_id:
        branch = data_store.get_branch(conversation_id, node_id)
        return {"branch": [node.dict() for node in branch]}
    
    # If no node_id, return all nodes organized by branches
    # This is a more complex operation since we need to construct the tree
    return {"branches": build_branch_tree(conversation)}

def build_branch_tree(conversation: Conversation):
    """Build a tree structure of branches from a conversation."""
    # Map of parent_id -> list of child nodes
    children_map = {}
    
    # Initialize empty list for each possible parent
    for node_id in conversation.nodes:
        children_map[node_id] = []
    
    # Populate children lists
    for node_id, node in conversation.nodes.items():
        if node.parent_id:
            children_map.setdefault(node.parent_id, []).append(node_id)
    
    # Now build the tree starting from root
    branches = []
    
    def build_tree(node_id, current_branch):
        current_branch.append(conversation.nodes[node_id].dict())
        
        # Process children
        children = children_map.get(node_id, [])
        
        if not children:
            # Leaf node, add this branch to results
            branches.append(list(current_branch))
        elif len(children) == 1:
            # Single child, continue this branch
            build_tree(children[0], current_branch)
        else:
            # Multiple children, branch for each
            for i, child_id in enumerate(children):
                if i == 0:
                    # Continue with the first child in the current branch
                    build_tree(child_id, current_branch)
                else:
                    # Create new branches for other children
                    new_branch = list(current_branch)
                    build_tree(child_id, new_branch)
    
    # Start from root if it exists
    if conversation.root_id:
        build_tree(conversation.root_id, [])
    
    return branches

@app.get("/api/conversations/{conversation_id}/nodes/{node_id}")
async def get_node(conversation_id: str, node_id: str):
    """Get a specific node."""
    node = data_store.get_node(conversation_id, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return {"node": node.dict()}

@app.delete("/api/conversations/{conversation_id}/nodes/{node_id}")
async def delete_node(conversation_id: str, node_id: str):
    """Delete a node and its children."""
    success = data_store.delete_node(conversation_id, node_id)
    if not success:
        raise HTTPException(status_code=404, detail="Node not found")
    
    # Notify all clients in the conversation
    await sio.emit('node:delete', {
        "conversation_id": conversation_id,
        "node_id": node_id
    }, room=f"conversation:{conversation_id}")
    
    return {"status": "success"}

@app.get("/api/users/active")
async def get_active_users(conversation_id: Optional[str] = Query(None)):
    """Get active users, optionally filtered by conversation."""
    active_users = data_store.get_active_users(conversation_id)
    return {"active_users": [user.dict() for user in active_users]}

# Use Socket.IO app as the main app with the FastAPI app inside
# To be imported by other modules
application = socket_app

# Background task to periodically save data
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(periodic_save())

async def periodic_save():
    """Periodically save all data to disk."""
    while True:
        try:
            data_store.save_all_if_needed()
        except Exception as e:
            logger.error(f"Error in periodic save: {str(e)}")
        await asyncio.sleep(15)  # Check every 15 seconds

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tree_conversation_backend:application", host="0.0.0.0", port=8000, reload=True)