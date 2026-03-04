"""
Abstract base class for embedding providers.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    """
    Abstract base class for text embedding providers.

    All embedding implementations must inherit from this class
    and implement the required methods.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Return the dimension of the embedding vectors.

        Returns:
            Integer dimension of embeddings
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Return the name of the embedding model.

        Returns:
            Model name string
        """
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Embed multiple texts efficiently in batches.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once

        Returns:
            List of embedding vectors
        """
        pass

    def embed_with_metadata(self, text: str, metadata: dict = None) -> dict:
        """
        Embed text and return with metadata.

        Args:
            text: Text to embed
            metadata: Optional metadata to include

        Returns:
            Dictionary with embedding and metadata
        """
        embedding = self.embed_text(text)
        return {
            "text": text,
            "embedding": embedding,
            "model": self.model_name,
            "dimension": self.dimension,
            "metadata": metadata or {}
        }
