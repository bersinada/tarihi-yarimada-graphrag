"""
Hybrid retriever combining vector similarity and graph traversal.

Uses Reciprocal Rank Fusion (RRF) to combine results from both sources.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .vector_retriever import VectorRetriever, VectorSearchResult
from .graph_retriever import GraphRetriever, GraphSearchResult

logger = logging.getLogger(__name__)


class RetrievalSource(Enum):
    """Source of retrieval result."""
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


@dataclass
class HybridSearchResult:
    """Combined result from hybrid retrieval."""
    entity: str
    label: str
    content: str
    vector_score: Optional[float]
    graph_score: Optional[float]
    combined_score: float
    source: RetrievalSource
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entity": self.entity,
            "label": self.label,
            "content": self.content,
            "vector_score": self.vector_score,
            "graph_score": self.graph_score,
            "combined_score": self.combined_score,
            "source": self.source.value,
            "metadata": self.metadata
        }


class HybridRetriever:
    """
    Combines vector similarity search with graph traversal.

    Implements Reciprocal Rank Fusion (RRF) for score combination:
    RRF_score = sum(1 / (k + rank)) for each result list

    The alpha parameter controls the balance:
    - alpha = 0: Vector only
    - alpha = 1: Graph only
    - alpha = 0.5: Equal weight

    Example:
        >>> hybrid = HybridRetriever(vector_retriever, graph_retriever)
        >>> results = hybrid.retrieve(
        ...     "Dikilitaşı Mısır'dan kim getirtti?",
        ...     entities=["Dikilitaş"],
        ...     intent="origin"
        ... )
    """

    def __init__(self,
                 vector_retriever: VectorRetriever,
                 graph_retriever: GraphRetriever,
                 alpha: float = 0.5,
                 rrf_k: int = 60,
                 semantic_threshold: float = 0.0,
                 max_graph_contexts: int = 15,
                 max_vector_results: int = 5):
        """
        Initialize hybrid retriever.

        Args:
            vector_retriever: Vector similarity search component
            graph_retriever: Graph traversal component
            alpha: Balance factor (0=vector only, 1=graph only)
            rrf_k: RRF constant (higher = more weight to lower ranks)
            semantic_threshold: Minimum combined score to include in results
            max_graph_contexts: Maximum graph contexts per entity
            max_vector_results: Maximum vector results to include
        """
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.semantic_threshold = semantic_threshold
        self.max_graph_contexts = max_graph_contexts
        self.max_vector_results = max_vector_results

    def retrieve(self,
                 query: str,
                 entities: Optional[List[str]] = None,
                 intent: Optional[str] = None,
                 top_k: int = 20) -> List[HybridSearchResult]:
        """
        Perform hybrid retrieval combining vector and graph search.

        Args:
            query: Natural language query
            entities: Extracted entities from query
            intent: Query intent (factual, relational, spatial, origin, etc.)
            top_k: Maximum results to return

        Returns:
            Sorted list of hybrid results
        """
        entities = entities or []

        # 1. Vector search (always performed)
        logger.info(f"Performing vector search for: {query}")
        vector_results = self.vector_retriever.search(query)
        logger.info(f"Vector search returned {len(vector_results)} results")

        # 2. Graph search (strategy depends on intent and entities)
        logger.info(f"Performing graph search with intent: {intent}, entities: {entities}")
        graph_results = self._perform_graph_search(entities, intent)
        logger.info(f"Graph search returned {len(graph_results)} results")

        # 3. Fuse results using RRF
        fused_results = self._fuse_results(vector_results, graph_results)
        logger.info(f"Fusion produced {len(fused_results)} combined results")

        # 4. Apply semantic threshold — drop low-score results
        if self.semantic_threshold > 0:
            before = len(fused_results)
            fused_results = [
                r for r in fused_results
                if r.combined_score >= self.semantic_threshold
                or r.source == RetrievalSource.GRAPH  # always keep direct graph hits
            ]
            dropped = before - len(fused_results)
            if dropped > 0:
                logger.info(f"Semantic threshold dropped {dropped} low-score results")

        return fused_results[:top_k]

    def _perform_graph_search(self,
                              entities: List[str],
                              intent: Optional[str]) -> List[GraphSearchResult]:
        """
        Select and execute appropriate graph search strategies.

        Args:
            entities: Extracted entity names
            intent: Query intent

        Returns:
            Combined graph search results
        """
        results = []

        if not entities:
            return results

        for entity in entities:
            # Always get entity context
            try:
                context = self.graph_retriever.get_entity_context(entity)
                results.extend(context)
            except Exception as e:
                logger.warning(f"Failed to get context for {entity}: {e}")

            # Intent-specific searches
            if intent == "spatial":
                try:
                    nearby = self.graph_retriever.get_nearby_structures(entity)
                    results.extend(nearby)
                except Exception as e:
                    logger.warning(f"Spatial search failed for {entity}: {e}")

            elif intent in ("relational", "factual"):
                try:
                    builders = self.graph_retriever.get_structure_builders(entity)
                    results.extend(builders)
                except Exception as e:
                    logger.warning(f"Builder search failed for {entity}: {e}")

                # Check for student queries
                try:
                    students = self.graph_retriever.get_person_students(entity)
                    results.extend(students)
                except Exception as e:
                    logger.debug(f"Student search failed for {entity}: {e}")

            elif intent == "origin":
                try:
                    origin = self.graph_retriever.trace_origin(entity)
                    results.extend(origin)
                except Exception as e:
                    logger.warning(f"Origin tracing failed for {entity}: {e}")

            elif intent == "comparative":
                # For comparative queries, get multi-hop for comparison
                try:
                    multi_hop = self.graph_retriever.multi_hop_search(entity, max_hops=2)
                    results.extend(multi_hop)
                except Exception as e:
                    logger.warning(f"Multi-hop search failed for {entity}: {e}")

        return results

    def _fuse_results(self,
                      vector_results: List[VectorSearchResult],
                      graph_results: List[GraphSearchResult]) -> List[HybridSearchResult]:
        """
        Fuse vector and graph results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) for each result list
        Combined with alpha weighting for vector vs graph preference.

        Args:
            vector_results: Results from vector search
            graph_results: Results from graph search

        Returns:
            Fused and ranked results
        """
        entity_data: Dict[str, Dict[str, Any]] = {}

        # Process vector results
        for rank, result in enumerate(vector_results, 1):
            # Use text or properties.id as entity key
            entity = result.text or result.properties.get("id", f"vector_{rank}")

            if entity not in entity_data:
                entity_data[entity] = {
                    "vector_rank": None,
                    "graph_rank": None,
                    "vector_score": None,
                    "label": result.label,
                    "properties": result.properties,
                    "graph_contexts": [],
                    "content": result.text
                }

            entity_data[entity]["vector_rank"] = rank
            entity_data[entity]["vector_score"] = result.score

            # Update content from document if more detailed
            if result.label == "Document" and result.properties.get("content"):
                entity_data[entity]["content"] = result.properties["content"]

        # Process graph results
        seen_graph_entities = {}
        for result in graph_results:
            entity = result.source_entity

            if entity not in entity_data:
                entity_data[entity] = {
                    "vector_rank": None,
                    "graph_rank": None,
                    "vector_score": None,
                    "label": "Entity",
                    "properties": result.properties,
                    "graph_contexts": [],
                    "content": ""
                }

            # Assign graph rank (first occurrence)
            if entity not in seen_graph_entities:
                seen_graph_entities[entity] = len(seen_graph_entities) + 1
                entity_data[entity]["graph_rank"] = seen_graph_entities[entity]

            # Add graph context
            entity_data[entity]["graph_contexts"].append(result.context)

            # Also add target entities as separate results
            target = result.target_entity
            if target and target not in entity_data:
                entity_data[target] = {
                    "vector_rank": None,
                    "graph_rank": len(seen_graph_entities) + 1,
                    "vector_score": None,
                    "label": result.properties.get("target_label", "Entity"),
                    "properties": {},
                    "graph_contexts": [result.context],
                    "content": ""
                }
                seen_graph_entities[target] = len(seen_graph_entities) + 1

        # Calculate RRF scores
        fused = []
        for entity, data in entity_data.items():
            rrf_score = 0.0

            # Vector contribution (weighted by 1 - alpha)
            if data["vector_rank"]:
                vector_rrf = 1.0 / (self.rrf_k + data["vector_rank"])
                rrf_score += (1 - self.alpha) * vector_rrf

            # Graph contribution (weighted by alpha)
            if data["graph_rank"]:
                graph_rrf = 1.0 / (self.rrf_k + data["graph_rank"])
                rrf_score += self.alpha * graph_rrf

            # Determine source
            if data["vector_rank"] and data["graph_rank"]:
                source = RetrievalSource.HYBRID
            elif data["vector_rank"]:
                source = RetrievalSource.VECTOR
            else:
                source = RetrievalSource.GRAPH

            # Build content string
            content_parts = []
            if data["content"]:
                content_parts.append(data["content"][:300])
            if data["graph_contexts"]:
                # Deduplicate and limit contexts
                unique_contexts = list(dict.fromkeys(data["graph_contexts"]))[:5]
                content_parts.append("İlişkiler: " + "; ".join(unique_contexts))

            # Calculate normalized graph score
            graph_score = None
            if data["graph_rank"]:
                graph_score = 1.0 / data["graph_rank"]

            fused.append(HybridSearchResult(
                entity=entity,
                label=data["label"],
                content="\n".join(content_parts),
                vector_score=data["vector_score"],
                graph_score=graph_score,
                combined_score=rrf_score,
                source=source,
                metadata={
                    "properties": data["properties"],
                    "graph_contexts": data["graph_contexts"][:10],
                    "vector_rank": data["vector_rank"],
                    "graph_rank": data["graph_rank"]
                }
            ))

        # Sort by combined score
        fused.sort(key=lambda x: x.combined_score, reverse=True)

        return fused

    def set_alpha(self, alpha: float) -> None:
        """
        Adjust the vector/graph balance dynamically.

        Args:
            alpha: New alpha value (0-1)
        """
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
        self.alpha = alpha
        logger.info(f"Hybrid retriever alpha set to {alpha}")
