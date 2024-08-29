from typing import List, Dict, Any
from abc import ABC, abstractmethod


class BaseDocumentStore(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 1000) -> List[str]:
        """
        Add documents to the document store and return the list of embedding IDs.

        Args:
            documents (List[Dict[str, Any]]): List of documents to add.
            batch_size (int, optional): Number of documents to process in each batch. Defaults to 1000.

        Returns:
            List[str]: List of IDs for the added embeddings.
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int = 4):
        """
        Search the document store for the most similar documents to the query.

        Args:
            query (str): The search query.
            k (int, optional): Number of top results to return. Defaults to 4.

        Returns:
            List[Dict[str, Any]]: List of search results with their metadata.
        """
        pass

    @abstractmethod
    def get_retriever(self, **kwargs):
        """
        Get a retriever object for the document store.

        Args:
            **kwargs: Additional arguments to pass to the retriever.

        Returns:
            Retriever: A retriever object for the document store.
        """
        pass

    @abstractmethod
    def save(self):
        """
        Save the document store state to a persistent storage.
        """
        pass

    @abstractmethod
    def remove_embeddings_by_ids(self, embedding_ids: List[str]):
        """
        Remove embeddings from the document store using their IDs.

        Args:
            embedding_ids (List[str]): List of IDs for the embeddings to be removed.
        """
        pass
