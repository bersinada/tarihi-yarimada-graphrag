"""
Neo4j Vector Index management for the GraphRAG system.

Handles creation of vector indexes and embedding of nodes.
"""

import logging
from typing import Any, Dict, List, Optional

from ..database.neo4j_client import Neo4jClient
from ..embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)


class VectorIndexManager:
    """
    Manages Neo4j vector indexes for semantic search.

    Responsibilities:
    - Create vector indexes for node labels
    - Embed existing nodes with their text properties
    - Update embeddings when nodes change

    Example:
        >>> manager = VectorIndexManager(client, embedder, dimension=384)
        >>> manager.create_indexes(["Yapi", "Document"])
        >>> manager.embed_nodes("Yapi")
    """

    # Default labels to index (matching Neo4j schema)
    DEFAULT_LABELS = [
        "Structure", "Building", "Monument", "Person",
        "Location", "City", "Region", "Street",
        "Event", "Organization", "Group", "Deity", "Document"
    ]

    # Properties to combine for embedding text
    TEXT_PROPERTIES = ["id", "name", "description", "content", "text"]

    def __init__(self,
                 client: Neo4jClient,
                 embedder: BaseEmbedder,
                 dimension: int = 384,
                 similarity_function: str = "cosine"):
        """
        Initialize the vector index manager.

        Args:
            client: Neo4j client instance
            embedder: Embedding provider
            dimension: Vector dimension (must match embedder)
            similarity_function: Similarity metric (cosine, euclidean, or dot)
        """
        self.client = client
        self.embedder = embedder
        self.dimension = dimension
        self.similarity_function = similarity_function

        # Verify dimensions match
        if embedder.dimension != dimension:
            raise ValueError(
                f"Embedder dimension ({embedder.dimension}) does not match "
                f"specified dimension ({dimension})"
            )

    def create_index(self, label: str) -> bool:
        """
        Create a vector index for a specific node label.

        Args:
            label: Node label (e.g., "Structure", "Document")

        Returns:
            True if index created or already exists
        """
        index_name = f"{label.lower()}_embedding"

        # Try modern syntax first (Neo4j 5.13+)
        modern_query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (n:{label})
        ON n.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.dimension},
                `vector.similarity_function`: '{self.similarity_function}'
            }}
        }}
        """

        try:
            self.client.execute_write(modern_query)
            logger.info(f"Created vector index: {index_name}")
            return True
        except Exception as e:
            # Fall back to procedure syntax (Neo4j 5.11-5.12)
            logger.debug(f"Modern syntax failed, trying procedure: {e}")

            procedure_query = f"""
            CALL db.index.vector.createNodeIndex(
                '{index_name}',
                '{label}',
                'embedding',
                {self.dimension},
                '{self.similarity_function}'
            )
            """
            try:
                self.client.execute_write(procedure_query)
                logger.info(f"Created vector index (procedure): {index_name}")
                return True
            except Exception as e2:
                logger.error(f"Failed to create index {index_name}: {e2}")
                return False

    def create_indexes(self, labels: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Create vector indexes for multiple labels.

        Args:
            labels: List of node labels (defaults to DEFAULT_LABELS)

        Returns:
            Dictionary mapping label to success status
        """
        labels = labels or self.DEFAULT_LABELS
        results = {}

        for label in labels:
            results[label] = self.create_index(label)

        return results

    def drop_index(self, label: str) -> bool:
        """
        Drop a vector index.

        Args:
            label: Node label

        Returns:
            True if dropped successfully
        """
        index_name = f"{label.lower()}_embedding"
        query = f"DROP INDEX {index_name} IF EXISTS"

        try:
            self.client.execute_write(query)
            logger.info(f"Dropped vector index: {index_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to drop index {index_name}: {e}")
            return False

    def embed_nodes(self,
                    label: str,
                    batch_size: int = 50,
                    force: bool = False) -> int:
        """
        Generate and store embeddings for all nodes of a label.

        Args:
            label: Node label to embed
            batch_size: Number of nodes to process at once
            force: If True, re-embed nodes that already have embeddings

        Returns:
            Number of nodes embedded
        """
        # Fetch nodes (optionally without existing embeddings)
        if force:
            fetch_query = f"""
            MATCH (n:{label})
            RETURN elementId(n) as node_id, properties(n) as props
            """
        else:
            fetch_query = f"""
            MATCH (n:{label})
            WHERE n.embedding IS NULL
            RETURN elementId(n) as node_id, properties(n) as props
            """

        nodes = self.client.execute_query(fetch_query)

        if not nodes:
            logger.info(f"No nodes to embed for label: {label}")
            return 0

        logger.info(f"Embedding {len(nodes)} nodes of type {label}")

        # Process in batches
        total_embedded = 0
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]

            # Build embedding texts
            texts = [self._build_embedding_text(n["props"]) for n in batch]
            node_ids = [n["node_id"] for n in batch]

            # Generate embeddings
            embeddings = self.embedder.embed_batch(texts)

            # Update nodes with embeddings
            for node_id, embedding in zip(node_ids, embeddings):
                update_query = """
                MATCH (n)
                WHERE elementId(n) = $node_id
                SET n.embedding = $embedding
                """
                self.client.execute_write(update_query, {
                    "node_id": node_id,
                    "embedding": embedding
                })

            total_embedded += len(batch)
            logger.info(f"Embedded {total_embedded}/{len(nodes)} {label} nodes")

        return total_embedded

    def embed_all_labels(self,
                         labels: Optional[List[str]] = None,
                         force: bool = False) -> Dict[str, int]:
        """
        Embed nodes for all specified labels.

        Args:
            labels: List of labels to embed (defaults to DEFAULT_LABELS)
            force: If True, re-embed existing embeddings

        Returns:
            Dictionary mapping label to count of embedded nodes
        """
        labels = labels or self.DEFAULT_LABELS
        results = {}

        for label in labels:
            try:
                count = self.embed_nodes(label, force=force)
                results[label] = count
            except Exception as e:
                logger.error(f"Failed to embed {label}: {e}")
                results[label] = -1

        return results

    def _build_embedding_text(self, props: Dict[str, Any]) -> str:
        """
        Build text representation from node properties for embedding.

        Combines relevant text properties into a single string.

        Args:
            props: Node properties dictionary

        Returns:
            Combined text for embedding
        """
        parts = []

        for prop in self.TEXT_PROPERTIES:
            if prop in props and props[prop]:
                value = props[prop]
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())

        # Also include any other string properties that might be relevant
        for key, value in props.items():
            if key not in self.TEXT_PROPERTIES and key != "embedding":
                if isinstance(value, str) and value.strip() and len(value) > 3:
                    parts.append(f"{key}: {value}")

        return " | ".join(parts) if parts else "unknown"

    def get_index_status(self) -> List[Dict[str, Any]]:
        """
        Get status of all vector indexes.

        Returns:
            List of index information dictionaries
        """
        query = """
        SHOW INDEXES
        WHERE type = 'VECTOR'
        """
        return self.client.execute_query(query)

    def verify_index_exists(self, label: str) -> bool:
        """
        Check if a vector index exists for a label.

        Args:
            label: Node label

        Returns:
            True if index exists
        """
        index_name = f"{label.lower()}_embedding"
        indexes = self.get_index_status()
        return any(idx.get("name") == index_name for idx in indexes)
