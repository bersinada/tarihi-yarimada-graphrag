"""
Sentence Transformer embedding implementation with Turkish language support.
"""

import logging
from typing import List, Optional

from .base import BaseEmbedder

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Embedding provider using Sentence Transformers.

    Uses the paraphrase-multilingual-MiniLM-L12-v2 model by default,
    which provides excellent Turkish language support with 384 dimensions.

    Features:
    - Local inference (no API calls)
    - Free to use
    - Fast inference
    - Native Turkish support

    Example:
        >>> embedder = SentenceTransformerEmbedder()
        >>> embedding = embedder.embed_text("Ayasofya tarihi bir yapidir")
        >>> len(embedding)
        384
    """

    # Default model with Turkish support
    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the Sentence Transformer embedder.

        Args:
            model_name: Name of the sentence-transformers model to use
            device: Device to run on ('cpu', 'cuda', or None for auto)
        """
        self._model_name = model_name or self.DEFAULT_MODEL

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading Sentence Transformer model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name, device=device)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Dimension: {self._dimension}")

        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: "
                "pip install sentence-transformers"
            )

    @property
    def dimension(self) -> int:
        """Return embedding dimension (384 for default model)."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: Turkish or English text to embed

        Returns:
            384-dimensional embedding vector
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * self._dimension

        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        return embedding.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Embed multiple texts efficiently.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Handle empty strings
        processed_texts = [t if t and t.strip() else " " for t in texts]

        embeddings = self._model.encode(
            processed_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100  # Show progress for large batches
        )

        return embeddings.tolist()

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        from sentence_transformers import util

        emb1 = self._model.encode(text1, convert_to_tensor=True)
        emb2 = self._model.encode(text2, convert_to_tensor=True)

        return util.cos_sim(emb1, emb2).item()
