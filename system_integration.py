import os
import asyncio
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def integrate_tree_conversation_with_coordinator(app: FastAPI, coordinator_url: str = "http://localhost:5010"):
    """
    Integrate the tree conversation system with an existing FastAPI app.
    
    Args:
        app: The FastAPI app to integrate with
        coordinator_url: URL of the coordinator service
    """
    from tree_conversation_backend import app as tree_app, data_store
    from coordinator_connector import integrate_with_tree_backend
    
    # Mount tree conversation API routes
    app.mount("/tree-api", tree_app)
    
    # Mount static files for the UI
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Serve tree_conversation_ui.html
    ui_path = os.path.join(current_dir, "tree_conversation_ui.html")
    if not os.path.exists(ui_path):
        logger.warning(f"UI file not found at {ui_path}. The UI will not be available.")
    else:
        # Create a static files directory if it doesn't exist
        static_dir = os.path.join(current_dir, "static")
        os.makedirs(static_dir, exist_ok=True)
        
        # Copy the UI file to the static directory
        with open(ui_path, "r") as f:
            ui_content = f.read()
        
        with open(os.path.join(static_dir, "tree_conversation_ui.html"), "w") as f:
            f.write(ui_content)
        
        # Mount the static directory
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    # Connect to the coordinator
    await integrate_with_tree_backend(tree_app, data_store)
    
    # Add a redirect route
    @app.get("/tree-ui")
    async def redirect_to_ui():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/static/tree_conversation_ui.html")
    
    logger.info("Tree conversation system integrated successfully")
    return app

def run_integration(host="0.0.0.0", port=8000, coordinator_url="http://localhost:5010"):
    """
    Run the integrated system with a new FastAPI app.
    """
    from fastapi import FastAPI
    import uvicorn
    
    app = FastAPI(title="EasyLM Integrated System")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with actual origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        return {"message": "EasyLM Integrated System is running", 
                "tree_ui_url": "/tree-ui"}
    
    # Run the integration
    asyncio.run(integrate_tree_conversation_with_coordinator(app, coordinator_url))
    
    # Start the server
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the integrated EasyLM system")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--coordinator-url", default="http://localhost:5010", 
                       help="URL of the coordinator service")
    
    args = parser.parse_args()
    
    run_integration(args.host, args.port, args.coordinator_url)