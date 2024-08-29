"""Main entry point for the FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langsmith import Client

from services.rag_service import answer_chain, ChatRequest
from langserve import add_routes
from api.routes import router as api_router

# Initialize the LangSmith client for logging and tracing
client = Client()

# Create an instance of the FastAPI app
app = FastAPI()

# Add CORS middleware to the app to handle cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,  # Allow credentials to be included in requests
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers in requests
    expose_headers=["*"],  # Expose all headers to the client
)

# Add routes to the app for the chat endpoint, using the answer_chain service
add_routes(
    app,
    answer_chain,
    path="/chat",  # Define the path for the chat endpoint
    input_type=ChatRequest,  # Define the input type for the chat endpoint
    config_keys=["metadata", "configurable", "tags"],  # Define the configuration keys for the chat endpoint
)

# Include additional API routes from the api_router module
app.include_router(api_router)

if __name__ == "__main__":
    # Run the FastAPI app using Uvicorn when the script is executed directly
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)