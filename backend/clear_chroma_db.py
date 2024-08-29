import os
from document_store.chroma_store import ChromaDocumentStore
from constants import COLLECTION_NAME
from services.ingest_service import get_embeddings_model


def clear_chroma_db():
    # Initialize the Chroma document store
    document_store = ChromaDocumentStore(
        collection_name=COLLECTION_NAME,
        embedding_model=get_embeddings_model(),
        index_path="chroma_index",
    )

    # Clear the Chroma index
    document_store.clear()
    print("Chroma DB cleared successfully.")


if __name__ == "__main__":
    clear_chroma_db()