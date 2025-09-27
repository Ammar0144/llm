#!/usr/bin/env python3
"""
Startup script for the DistilGPT-2 LLM Server
"""

import uvicorn
from app import app

if __name__ == "__main__":
    print("Starting DistilGPT-2 LLM Server...")
    print("Server will be available at: http://localhost:8082")
    print("API documentation at: http://localhost:8082/docs")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8082,
        log_level="info"
    )
# to deploy 2