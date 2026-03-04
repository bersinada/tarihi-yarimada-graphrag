"""
GraphRAG - Hybrid Graph + Vector Retrieval Augmented Generation
Istanbul Historical Peninsula Knowledge Graph System
"""

from .facade import GraphRAGFacade, QueryResult

__version__ = "1.0.0"
__all__ = ["GraphRAGFacade", "QueryResult"]


def ask(query: str, config_path: str = "config.yaml") -> str:
    """
    Quick query function for simple usage.

    Args:
        query: Natural language question in Turkish
        config_path: Path to configuration file

    Returns:
        Generated response string

    Example:
        >>> from graphrag import ask
        >>> answer = ask("Ayasofya'yi kim yaptirdi?")
    """
    with GraphRAGFacade(config_path) as rag:
        result = rag.query(query)
        return result.response
