"""Embedding providers for vector representation of text."""

from .base import BaseEmbedder
from .sentence_transformer import SentenceTransformerEmbedder


def get_embedder(provider: str = "sentence_transformer", model: str = None) -> BaseEmbedder:
    """
    Factory function to get an embedder instance.

    Args:
        provider: Embedding provider name ("sentence_transformer" or "openai")
        model: Optional model name override

    Returns:
        BaseEmbedder instance
    """
    if provider == "sentence_transformer":
        model = model or "paraphrase-multilingual-MiniLM-L12-v2"
        return SentenceTransformerEmbedder(model_name=model)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


__all__ = ["BaseEmbedder", "SentenceTransformerEmbedder", "get_embedder"]
