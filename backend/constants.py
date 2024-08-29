import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Fetch environment variables from the environment or .env file
DATABASE_HOST = os.getenv("DATABASE_HOST")  # The hostname or IP address of the database server
DATABASE_PORT = os.getenv("DATABASE_PORT")  # The port number on which the database server is listening
DATABASE_USERNAME = os.getenv("DATABASE_USERNAME")  # The username to connect to the database
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD")  # The password to connect to the database
DATABASE_NAME = os.getenv("DATABASE_NAME")  # The name of the database to connect to
COLLECTION_NAME = os.getenv("COLLECTION_NAME")  # The name of the collection used in the document store
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # API token for Hugging Face Hub
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # API key for OpenAI

# Construct the database URL for SQLAlchemy using the fetched environment variables
RECORD_MANAGER_DB_URL = (
    f"postgresql://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
)
