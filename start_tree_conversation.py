#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import webbrowser
import time
import logging
import asyncio

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TreeConversation")

def parse_args():
    parser = argparse.ArgumentParser(description="Start the EasyLM Tree Conversation system")
    parser.add_argument("--coordinator-url", type=str,
                      help="URL of the coordinator service (default: http://51.81.181.136:5010)")
    parser.add_argument("--use-local-coordinator", action="store_true",
                      help="Start a local coordinator instead of using the remote one")
    parser.add_argument("--coordinator-port", type=int, default=5010,
                      help="Port for the local coordinator service (default: 5010)")
    parser.add_argument("--backend-port", type=int, default=8000,
                      help="Port for the backend service (default: 8000)")
    parser.add_argument("--no-open-browser", action="store_true",
                      help="Don't automatically open the UI in a browser")
    parser.add_argument("--ui-only", action="store_true",
                      help="Start only the UI (assuming backend is already running)")
    parser.add_argument("--data-dir", type=str, default="conversation_data",
                      help="Directory to store conversation data (default: conversation_data)")
    return parser.parse_args()

async def start_backend(args):
    """Start the tree conversation backend."""
    logger.info(f"Starting tree conversation backend on port {args.backend_port}...")
    logger.info(f"Connecting to coordinator at {args.coordinator_url}")
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Set environment variables
    env = os.environ.copy()
    env["CONVERSATION_DATA_DIR"] = args.data_dir
    
    # Run the backend server
    cmd = [
        sys.executable, 
        "-m", 
        "coordinator_connector", 
        "--port", 
        str(args.backend_port), 
        "--coordinator-url", 
        args.coordinator_url
    ]
    
    try:
        process = subprocess.Popen(cmd, env=env)
        logger.info(f"Backend started with PID {process.pid}")
        return process
    except Exception as e:
        logger.error(f"Failed to start backend: {str(e)}")
        return None

def start_coordinator(args):
    """Start the EasyLM coordinator service."""
    logger.info(f"Starting EasyLM coordinator on port {args.coordinator_port}...")
    
    # Run the coordinator
    cmd = [
        sys.executable, 
        "-m", 
        "EasyLM.coordinator", 
        "--port", 
        str(args.coordinator_port)
    ]
    
    try:
        process = subprocess.Popen(cmd)
        logger.info(f"Coordinator started with PID {process.pid}")
        return process
    except Exception as e:
        logger.error(f"Failed to start coordinator: {str(e)}")
        return None

def open_ui(args):
    """Open the UI in a web browser."""
    url = f"http://localhost:{args.backend_port}/tree_conversation_ui.html"
    logger.info(f"Opening UI in browser: {url}")
    webbrowser.open(url)

async def serve_ui(args):
    """Serve the UI using a simple HTTP server."""
    # Create a temporary HTML file that redirects to the main UI
    with open("redirect.html", "w") as f:
        f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0;URL='http://localhost:{args.backend_port}/tree_conversation_ui.html'" />
</head>
<body>
    <p>Redirecting to <a href="http://localhost:{args.backend_port}/tree_conversation_ui.html">Tree Conversation UI</a>...</p>
</body>
</html>
        """)
    
    # Serve the current directory on a simple HTTP server
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    
    class MyHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=os.getcwd(), **kwargs)
    
    httpd = HTTPServer(("localhost", 8080), MyHandler)
    logger.info("Serving UI on http://localhost:8080/")
    
    if not args.no_open_browser:
        webbrowser.open("http://localhost:8080/redirect.html")
    
    httpd.serve_forever()

async def main():
    args = parse_args()
    
    # Default to the remote coordinator if not specified
    if args.coordinator_url is None:
        if args.use_local_coordinator:
            args.coordinator_url = f"http://localhost:{args.coordinator_port}"
            logger.info(f"Using local coordinator at {args.coordinator_url}")
        else:
            args.coordinator_url = "http://51.81.181.136:5010"
            logger.info(f"Using remote coordinator at {args.coordinator_url}")
    
    processes = []
    
    if not args.ui_only:
        # Start local coordinator only if explicitly requested
        if args.use_local_coordinator:
            coordinator = start_coordinator(args)
            if coordinator:
                processes.append(coordinator)
                
            # Wait for coordinator to start
            logger.info("Waiting for local coordinator to start...")
            time.sleep(5)
            args.coordinator_url = f"http://localhost:{args.coordinator_port}"
                
        # Start backend
        backend = await start_backend(args)
        if backend:
            processes.append(backend)
            
        # Wait for backend to start
        logger.info("Waiting for backend to start...")
        time.sleep(5)
    
    if not args.no_open_browser:
        open_ui(args)
    
    # Keep script running and handle shutdown
    try:
        logger.info(f"System is running with coordinator at {args.coordinator_url}. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping processes...")
        for process in processes:
            process.terminate()
        logger.info("All processes stopped.")

if __name__ == "__main__":
    asyncio.run(main())