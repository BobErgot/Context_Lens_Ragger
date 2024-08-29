"""Main entrypoint for the app."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langsmith import Client

from services.rag_service import answer_chain, ChatRequest
from langserve import add_routes
from api.routes import router as api_router

client = Client()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

add_routes(
    app,
    answer_chain,
    path="/chat",
    input_type=ChatRequest,
    config_keys=["metadata", "configurable", "tags"],
)

# Include the static API routes
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)