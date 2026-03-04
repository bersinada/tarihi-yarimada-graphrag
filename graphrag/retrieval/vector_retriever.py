"""
Vector-based semantic search retriever using Neo4j vector indexes.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..database.neo4j_client import Neo4jClient
from ..embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""
    node_id: str
    label: str
    properties: Dict[str, Any]
    score: float
    text: str
    source: str = "vector"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "label": self.label,
            "properties": self.properties,
            "score": self.score,
            "text": self.text,
            "source": self.source
        }


class VectorRetriever:
    """
    Semantic similarity search using Neo4j vector indexes.

    Searches both entity nodes (Yapi, Padisah, etc.) and
    Document chunks for relevant content.

    Example:
        >>> retriever = VectorRetriever(client, embedder)
        >>> results = retriever.search("Ayasofya hakkinda bilgi")
        >>> for r in results:
        ...     print(f"{r.label}: {r.text} (score: {r.score:.2f})")
    """

    # Labels to search by default
    DEFAULT_ENTITY_LABELS = ["Structure", "Building", "Monument", "Person", "Location", "Event"]
    DOCUMENT_LABEL = "Document"

    def __init__(self,
                 client: Neo4jClient,
                 embedder: BaseEmbedder,
                 top_k: int = 10,
                 min_score: float = 0.4):
        """
        Initialize the vector retriever.

        Args:
            client: Neo4j client instance
            embedder: Embedding provider
            top_k: Maximum results per label
            min_score: Minimum similarity threshold
        """
        self.client = client
        self.embedder = embedder
        self.top_k = top_k
        self.min_score = min_score

    def search(self,
               query: str,
               labels: Optional[List[str]] = None,
               include_documents: bool = True,
               top_k: Optional[int] = None,
               min_score: Optional[float] = None) -> List[VectorSearchResult]:
        """
        Perform vector similarity search across multiple labels.

        Args:
            query: Natural language query
            labels: Node labels to search (defaults to entity labels)
            include_documents: Whether to also search Document nodes
            top_k: Override default top_k
            min_score: Override default min_score

        Returns:
            Sorted list of search results (highest score first)
        """
        top_k = top_k or self.top_k
        min_score = min_score or self.min_score
        labels = labels or self.DEFAULT_ENTITY_LABELS

        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)

        all_results = []

        # Search entity labels
        for label in labels:
            try:
                results = self._search_label(
                    query_embedding, label, top_k, min_score
                )
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Vector search failed for {label}: {e}")

        # Search documents if requested
        if include_documents:
            try:
                doc_results = self._search_documents(
                    query_embedding, top_k, min_score
                )
                all_results.extend(doc_results)
            except Exception as e:
                logger.warning(f"Document vector search failed: {e}")

        # Sort by score and return top results
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k * 2]  # Return more to allow for diversity

    def _search_label(self,
                      embedding: List[float],
                      label: str,
                      top_k: int,
                      min_score: float) -> List[VectorSearchResult]:
        """
        Search a specific label's vector index.

        Args:
            embedding: Query embedding vector
            label: Node label to search
            top_k: Maximum results
            min_score: Minimum similarity

        Returns:
            List of search results
        """
        index_name = f"{label.lower()}_embedding"

        query = f"""
        CALL db.index.vector.queryNodes(
            '{index_name}',
            $top_k,
            $embedding
        ) YIELD node, score
        WHERE score >= $min_score
        RETURN
            elementId(node) as node_id,
            node.id as name,
            labels(node)[0] as label,
            properties(node) as props,
            score
        """

        try:
            results = self.client.execute_query(query, {
                "embedding": embedding,
                "top_k": top_k,
                "min_score": min_score
            })

            return [
                VectorSearchResult(
                    node_id=r["node_id"],
                    label=r["label"],
                    properties=self._clean_properties(r["props"]),
                    score=r["score"],
                    text=r["name"] or ""
                )
                for r in results
            ]
        except Exception as e:
            # Index might not exist
            if "index" in str(e).lower():
                logger.debug(f"Vector index {index_name} not found")
            else:
                logger.warning(f"Vector search error for {label}: {e}")
            return []

    def _search_documents(self,
                          embedding: List[float],
                          top_k: int,
                          min_score: float) -> List[VectorSearchResult]:
        """
        Search Document node vector index.

        Args:
            embedding: Query embedding vector
            top_k: Maximum results
            min_score: Minimum similarity

        Returns:
            List of document search results
        """
        query = """
        CALL db.index.vector.queryNodes(
            'document_embedding',
            $top_k,
            $embedding
        ) YIELD node, score
        WHERE score >= $min_score
        OPTIONAL MATCH (node)-[:DESCRIBES]->(s)
        WHERE s:Structure OR s:Building OR s:Monument
        RETURN
            elementId(node) as node_id,
            node.id as doc_id,
            node.content as content,
            node.source_file as source,
            s.id as related_structure,
            score
        """

        try:
            results = self.client.execute_query(query, {
                "embedding": embedding,
                "top_k": top_k,
                "min_score": min_score
            })

            return [
                VectorSearchResult(
                    node_id=r["node_id"],
                    label="Document",
                    properties={
                        "content": r["content"],
                        "source_file": r["source"],
                        "related_structure": r["related_structure"]
                    },
                    score=r["score"],
                    text=r["content"][:200] if r["content"] else "",
                    source="document"
                )
                for r in results
            ]
        except Exception as e:
            if "index" in str(e).lower():
                logger.debug("Document vector index not found")
            else:
                logger.warning(f"Document vector search error: {e}")
            return []

    def search_similar_to_entity(self,
                                 entity_name: str,
                                 label: str,
                                 top_k: int = 5) -> List[VectorSearchResult]:
        """
        Find entities similar to a given entity.

        Args:
            entity_name: Name of the entity to find similar items for
            label: Label of the entity
            top_k: Maximum results

        Returns:
            List of similar entities
        """
        # Get the entity's embedding
        query = f"""
        MATCH (n:{label})
        WHERE n.id = $name OR toLower(n.id) CONTAINS toLower($name)
        RETURN n.embedding as embedding, n.id as name
        LIMIT 1
        """

        result = self.client.execute_query(query, {"name": entity_name})

        if not result or not result[0].get("embedding"):
            logger.warning(f"No embedding found for entity: {entity_name}")
            return []

        embedding = result[0]["embedding"]

        # Search for similar entities (excluding the original)
        return self._search_label(embedding, label, top_k + 1, self.min_score)[1:]

    def _clean_properties(self, props: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean properties by removing embedding vectors.

        Args:
            props: Raw properties dictionary

        Returns:
            Cleaned properties
        """
        return {k: v for k, v in props.items() if k != "embedding"}
