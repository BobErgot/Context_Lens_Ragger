# Context_Lens_Ragger

This repository contains an implementation of a locally hosted chatbot focused on question answering and document analysis. The project is built using [LangChain](https://github.com/langchain-ai/langchain/), [FastAPI](https://fastapi.tiangolo.com/), and [Next.js](https://nextjs.org/), and leverages advanced document processing, retrieval, and language modeling techniques to provide accurate and context-aware answers from uploaded documents.

## Features
- **Document Processing**: Handles a variety of document formats including PDFs, DOCX, images (PNG, JPG, JPEG), and SVGs. It extracts text using Tesseract OCR and generates detailed image captions using BLIP. The service also supports the conversion of SVG files to PNG for further processing.
- **Comprehensive Descriptions**: Combines OCR-extracted text and image captions to generate detailed and comprehensive descriptions using OpenAI's GPT-3.5 model.
- **Chroma Document Store**: Utilizes Chroma for efficient document storage, retrieval, and embedding-based search. The document store supports the addition, querying, and removal of document embeddings.
- **API for Document Management**: Provides RESTful API endpoints for uploading, fetching, and deleting documents. Users can manage their document collection directly through the API.
- **Evaluation Chain**: Includes a chain setup that enables evaluation of model responses using custom datasets, leveraging LangSmith for advanced model performance assessment.
- **Environment Configuration**: Simplifies configuration through environment variables for setting up API keys, database connections, and tracing options. This makes the application flexible and easy to deploy in different environments.
- **Feedback Integration**: Supports feedback submission and updating, allowing users to provide input on the quality of generated responses for continuous improvement.

## Running locally

1. Install backend dependencies: `poetry install`.

2. Set up your environment variables by creating a `.env` file in the project root directory with the following content:

   ```dotenv
   # Database Configuration
   DATABASE_HOST="192.168.215.2"
   DATABASE_PORT="5432"
   DATABASE_USERNAME="username"
   DATABASE_PASSWORD="password"
   DATABASE_NAME="recordmanager"

   # Collection Name for Document Store
   COLLECTION_NAME="support_articles"

   # OpenAI API Key
   OPENAI_API_KEY="your-openai-api-key-here"

   # Record Manager Database URL
   RECORD_MANAGER_DB_URL="your-db-url"

   # LangChain Tracing Configuration
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
   LANGCHAIN_API_KEY="your-langchain-api-key-here"
   LANGCHAIN_PROJECT="your-project"

   # Hugging Face API Token
   HUGGINGFACEHUB_API_TOKEN="your-huggingfacehub-api-token-here"
   ```
3.	Upload, delete, and view documents directly from the Next.js application:
      •	The application provides a user-friendly interface to manage your documents. You can upload new documents, view existing ones, and delete them as needed.
4.	Start the Python backend with make start.
5.	Install frontend dependencies by running cd ./frontend, then yarn.
6.	Run the frontend with yarn dev.
7.	Open localhost:3000 in your browser to access the application.

## Database and Vector DB Setup

To run the application locally, we'll utilize **Ollama** for LLM inference and embeddings generation. For the vector store, we'll use **Chroma**, a free open-source vector store. For managing records, we'll use a simple **PostgreSQL** database. Additionally, to run Chroma and PostgreSQL, you'll need to have **Docker** installed.

### Steps

### Docker Installation

To download and manage Docker containers with a GUI, you can download **OrbStack** [here](https://orbstack.dev/). Once set up, we can proceed with installing Chroma and PostgreSQL.

### Chroma Setup

1. Clone the official Chroma repository:

   ```bash
    git clone git@github.com:chroma-core/chroma.git
   ```

2.	Navigate into the cloned repository and start the Docker container:
   ```bash
    cd chroma
    docker-compose up -d --build
   ```

### PostgreSQL Setup

1.	Pull the PostgreSQL image:
   ```bash
    docker pull postgres
   ```

2.	Start the PostgreSQL image with the following command:

   ```bash
    docker run --name postgres -e POSTGRES_PASSWORD=mysecretpassword -d postgres
   ```
This setup will allow you to run the Context_Lens_Ragger locally, enabling you to test and develop the application with your own documents and data.