import logging
import os
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from document_store.base import BaseDocumentStore

logger = logging.getLogger(__name__)


class ChromaDocumentStore(BaseDocumentStore):
    """
    Document store implementation which uses Chroma for generating and storing vector embeddings.
    """

    def __init__(self, collection_name: str, embedding_model: Embeddings, index_path: str = "chroma_index") -> None:
        """
        Initialize the ChromaDocumentStore

        Args:
            collection_name(str): Name of the Chroma collection.
            embedding_model(Embeddings): Embedding model to use.
            index_path(str): Path to save/load the Chroma index. Defaults to "chroma_index".
        """
        self.collection_name = collection_name
        self.embeddings = embedding_model
        self.index_path = index_path

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.index_path,
        )
        self.document_count = 0

    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 1000) -> List[str]:
        """
        Add documents to the Chroma index and return the list of embedding IDs.

        Args:
            documents (List[Dict[str, Any]]): List of documents to add.
            batch_size (int, optional): Number of documents to process in each batch. Defaults to 1000.

        Returns:
            List[str]: List of IDs for the added embeddings.
        """
        if not documents:
            logger.warning("Attempted to add empty document list.")
            return []

        embedding_ids = []
        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                texts = [doc["content"] for doc in batch]
                metadatas = [doc["metadata"] for doc in batch]

                ids = self.vector_store.add_texts(texts, metadatas=metadatas)
                embedding_ids.extend(ids)

            self.document_count += len(documents)
            self.save()

            logger.info(f"Successfully loaded {len(documents)} documents to the store.")
            return embedding_ids
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    def search(self, query: str, k: int = 4):
        if self.vector_store is None:
            logger.error("Attempted search on empty vector store.")
            raise ValueError("No documents have been added to the store yet.")

        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)

            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                }
                for doc, score in results
            ]

        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise

    def get_retriever(self, **kwargs):
        """
        Get a retriever object for the vector store.

        Args:
            **kwargs: Additional arguments to pass to the retriever.

        Returns:
            Retriever: A retriever object for the vector store.
        """
        try:
            if self.vector_store is None:
                logger.info("Initializing empty vector store for retriever.")
                self.vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=self.index_path,
                )
            return self.vector_store.as_retriever(**kwargs)
        except Exception as e:
            logger.error(f"Error getting retriever: {str(e)}")
            raise

    def save(self):
        """
        Save the Chroma index to the specified persist directory.
        """
        try:
            if self.vector_store is None:
                logger.error("Attempted to save empty vector store")
                raise ValueError("No documents have been added to the store yet.")
            logger.info(f"Vector store saved to {self.index_path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise

    def clear(self):
        """
        Clear the Chroma index and reset the document count.
        """
        if os.path.exists(self.index_path):
            for file in os.listdir(self.index_path):
                file_path = os.path.join(self.index_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.index_path,
        )
        self.document_count = 0
        logger.info("Vector store cleared.")
        self.save()

    def get_document_count(self) -> int:
        """
        Get the number of documents in the store.

        Returns:
            int: The number of documents in the store.
        """
        return self.document_count

    def remove_embeddings_by_ids(self, embedding_ids: List[str]):
        """
        Remove embeddings from the Chroma index using their IDs.

        Args:
            embedding_ids (List[str]): List of IDs for the embeddings to be removed.
        """
        if not embedding_ids:
            logger.warning("No embedding IDs provided for deletion.")
            return

        try:
            self.vector_store.delete(ids=embedding_ids)
            self.save()
            logger.info(f"Removed embeddings with IDs: {embedding_ids}")
        except Exception as e:
            logger.error(f"Error removing embeddings: {str(e)}")
            raise
