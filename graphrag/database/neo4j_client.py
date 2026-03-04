"""
Neo4j database client wrapper with connection management.
"""

import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from neo4j import GraphDatabase, Driver, Session, Result

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Neo4j database client with connection pooling and context management.

    Example:
        >>> client = Neo4jClient(uri, username, password)
        >>> with client.session() as session:
        ...     result = session.run("MATCH (n) RETURN n LIMIT 10")
        >>> client.close()
    """

    def __init__(self, uri: str, username: str, password: str):
        """
        Initialize Neo4j client.

        Args:
            uri: Neo4j connection URI (e.g., bolt://localhost:7687)
            username: Database username
            password: Database password
        """
        self.uri = uri
        self.username = username
        self._driver: Optional[Driver] = None

        try:
            self._driver = GraphDatabase.driver(
                uri,
                auth=(username, password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )
            # Verify connectivity
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    @property
    def driver(self) -> Driver:
        """Get the Neo4j driver instance."""
        if self._driver is None:
            raise RuntimeError("Neo4j client is not connected")
        return self._driver

    @contextmanager
    def session(self, database: str = "neo4j") -> Generator[Session, None, None]:
        """
        Create a session context manager.

        Args:
            database: Database name (default: neo4j)

        Yields:
            Neo4j session
        """
        session = self.driver.session(database=database)
        try:
            yield session
        finally:
            session.close()

    def execute_query(self,
                      query: str,
                      parameters: Optional[Dict[str, Any]] = None,
                      database: str = "neo4j") -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results as list of dictionaries.

        Args:
            query: Cypher query string
            parameters: Query parameters
            database: Database name

        Returns:
            List of result records as dictionaries
        """
        with self.session(database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def execute_write(self,
                      query: str,
                      parameters: Optional[Dict[str, Any]] = None,
                      database: str = "neo4j") -> Dict[str, Any]:
        """
        Execute a write query within a transaction.

        Args:
            query: Cypher query string
            parameters: Query parameters
            database: Database name

        Returns:
            Query counters/summary
        """
        with self.session(database) as session:
            result = session.run(query, parameters or {})
            summary = result.consume()
            return {
                "nodes_created": summary.counters.nodes_created,
                "nodes_deleted": summary.counters.nodes_deleted,
                "relationships_created": summary.counters.relationships_created,
                "relationships_deleted": summary.counters.relationships_deleted,
                "properties_set": summary.counters.properties_set
            }

    def get_node_labels(self) -> List[str]:
        """Get all node labels in the database."""
        result = self.execute_query("CALL db.labels() YIELD label RETURN label")
        return [r["label"] for r in result]

    def get_relationship_types(self) -> List[str]:
        """Get all relationship types in the database."""
        result = self.execute_query("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
        return [r["relationshipType"] for r in result]

    def get_node_count(self, label: Optional[str] = None) -> int:
        """
        Get count of nodes, optionally filtered by label.

        Args:
            label: Optional node label filter

        Returns:
            Number of nodes
        """
        if label:
            query = f"MATCH (n:{label}) RETURN count(n) as count"
        else:
            query = "MATCH (n) RETURN count(n) as count"

        result = self.execute_query(query)
        return result[0]["count"] if result else 0

    def get_indexes(self) -> List[Dict[str, Any]]:
        """Get all indexes in the database."""
        return self.execute_query("SHOW INDEXES")

    def verify_vector_support(self) -> bool:
        """
        Check if the Neo4j version supports vector indexes.

        Returns:
            True if vector indexes are supported
        """
        try:
            result = self.execute_query("CALL dbms.components() YIELD versions RETURN versions[0] as version")
            if result:
                version = result[0]["version"]
                major, minor = map(int, version.split(".")[:2])
                # Vector indexes require Neo4j 5.11+
                return major > 5 or (major == 5 and minor >= 11)
        except Exception as e:
            logger.warning(f"Could not verify Neo4j version: {e}")
        return False

    def close(self):
        """Close the database connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
