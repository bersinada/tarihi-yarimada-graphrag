"""
Graph-based structured retriever using Cypher traversals.

Handles multi-hop relationships, spatial queries, and origin tracing.

Updated for English labels/relationships in Neo4j:
- Labels: Structure, Building, Monument, Person, Location, City, etc.
- Relations: COMMISSIONED_BY, DESIGNED_BY, LOCATED_IN, NEAR, etc.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from ..database.neo4j_client import Neo4jClient
from . import cypher_templates

logger = logging.getLogger(__name__)


@dataclass
class GraphSearchResult:
    """Result from graph traversal search."""
    source_entity: str
    target_entity: str
    relationship: str
    direction: str
    path_length: int
    properties: Dict[str, Any]
    context: str
    source: str = "graph"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "relationship": self.relationship,
            "direction": self.direction,
            "path_length": self.path_length,
            "properties": self.properties,
            "context": self.context,
            "source": self.source
        }


class GraphRetriever:
    """
    Structured graph traversal retriever.

    Provides various traversal strategies based on query intent:
    - Entity context (all relationships)
    - Origin tracing (for artifacts like Dikilitaş)
    - Spatial relationships (nearby structures)
    - Multi-hop exploration
    - Builder/architect relationships

    Example:
        >>> retriever = GraphRetriever(client)
        >>> context = retriever.get_entity_context("Ayasofya")
        >>> for r in context:
        ...     print(f"{r.source_entity} --{r.relationship}--> {r.target_entity}")
    """

    # English relationship type to Turkish description mapping
    RELATION_DESCRIPTIONS = {
        # Builder/Creator relationships
        "COMMISSIONED_BY": "tarafından yaptırıldı",
        "BUILT_BY": "tarafından inşa edildi",
        "DESIGNED_BY": "tarafından tasarlandı",
        "RESTORED_BY": "tarafından restore edildi",
        "ADDED_STRUCTURE": "yapı ekledi",
        "INSCRIBED_BY": "tarafından yazıldı",

        # Location relationships
        "LOCATED_IN": "konumundadır",
        "NEAR": "yakınındadır",

        # Origin/Transfer relationships
        "ORIGINALLY_LOCATED_IN": "orijinal olarak buradaydı",
        "TRANSFERRED_TO": "buraya taşındı",
        "ORIGINATED_FROM": "kökenli",

        # Person relationships
        "STUDENT_OF": "öğrencisidir",
        "PRESENTED_TO": "hediye edildi",
        "BURIAL_PLACE": "gömüldüğü yer",
        "TOMB_OF": "türbesi",

        # Historical/Event relationships
        "COMMEMORATES": "anısına yapıldı",
        "DEDICATED_TO": "adandı",
        "CONVERTED_TO_MOSQUE_BY": "tarafından camiye dönüştürüldü",
        "REPRESENTS": "temsil eder",
        "OPPOSES": "karşı tarafı temsil eder",

        # Document relationships
        "DESCRIBES": "hakkında bilgi verir"
    }

    def __init__(self, client: Neo4jClient, max_hops: int = 3):
        """
        Initialize the graph retriever.

        Args:
            client: Neo4j client instance
            max_hops: Maximum hops for multi-hop queries
        """
        self.client = client
        self.max_hops = max_hops

    def get_entity_context(self, entity_name: str) -> List[GraphSearchResult]:
        """
        Get all relationships and context for an entity.

        Args:
            entity_name: Name of the entity (e.g., "Ayasofya")
            is_spatial: Whether the query requires spatial data

        Returns:
            List of all relationships for the entity
        """
        results = self.client.execute_query(
            cypher_templates.ENTITY_FULL_CONTEXT,
            {"entity_name": entity_name}
        )

        if not results:
            logger.warning(f"No entity found: {entity_name}")
            return []

        entity_data = results[0]
        entity_id = entity_data.get("entity", {}).get("id", entity_name)
        relationships = entity_data.get("relationships", [])
        
        # Extract entity's own properties
        entity_props = entity_data.get("entity_props", {})

        graph_results = []
        for rel in relationships:
            if rel.get("related_id"):
                # Merge entity properties with relationship properties
                rel_props = rel.get("rel_props", {})
                combined_props = {**entity_props}
                combined_props.update(rel_props)
                
                graph_results.append(GraphSearchResult(
                    source_entity=entity_id,
                    target_entity=rel["related_id"],
                    relationship=rel["relation"],
                    direction=rel["direction"],
                    path_length=1,
                    properties=combined_props,
                    context=self._build_context_string(
                        entity_id,
                        rel["relation"],
                        rel["related_id"],
                        rel["direction"]
                    )
                ))

        return graph_results

    def trace_origin(self, artifact_name: str) -> List[GraphSearchResult]:
        """
        Trace the origin and journey of an artifact.

        Useful for queries like "Dikilitaşı Mısır'dan kim getirtti?"

        Args:
            artifact_name: Name of the artifact (e.g., "Dikilitaş")

        Returns:
            List of origin-related relationships
        """
        results = self.client.execute_query(
            cypher_templates.TRACE_ORIGIN,
            {"artifact_name": artifact_name}
        )

        if not results:
            return []

        data = results[0]
        graph_results = []
        artifact_id = data.get("artifact", artifact_name)

        # Add original location
        if data.get("original_location"):
            graph_results.append(GraphSearchResult(
                source_entity=artifact_id,
                target_entity=data["original_location"],
                relationship="ORIGINALLY_LOCATED_IN",
                direction="outgoing",
                path_length=1,
                properties={},
                context=f"{artifact_id} orijinal olarak {data['original_location']}'daydı"
            ))

        # Add origin region
        if data.get("origin_region"):
            graph_results.append(GraphSearchResult(
                source_entity=artifact_id,
                target_entity=data["origin_region"],
                relationship="ORIGINATED_FROM",
                direction="outgoing",
                path_length=1,
                properties={},
                context=f"{artifact_id} {data['origin_region']} kökenlidir"
            ))

        # Add commissioner (who brought it)
        if data.get("commissioned_by"):
            graph_results.append(GraphSearchResult(
                source_entity=artifact_id,
                target_entity=data["commissioned_by"],
                relationship="COMMISSIONED_BY",
                direction="outgoing",
                path_length=1,
                properties={"type": data.get("commissioner_type")},
                context=f"{artifact_id} {data['commissioned_by']} tarafından getirildi/yaptırıldı"
            ))

        # Add builder
        if data.get("built_by"):
            graph_results.append(GraphSearchResult(
                source_entity=artifact_id,
                target_entity=data["built_by"],
                relationship="BUILT_BY",
                direction="outgoing",
                path_length=1,
                properties={"type": data.get("builder_type")},
                context=f"{artifact_id} {data['built_by']} tarafından inşa edildi"
            ))

        # Add transfer destination
        if data.get("transferred_to"):
            graph_results.append(GraphSearchResult(
                source_entity=artifact_id,
                target_entity=data["transferred_to"],
                relationship="TRANSFERRED_TO",
                direction="outgoing",
                path_length=1,
                properties={},
                context=f"{artifact_id} {data['transferred_to']}'a taşındı"
            ))

        return graph_results

    def get_nearby_structures(self, structure_name: str) -> List[GraphSearchResult]:
        """
        Find spatially related structures.

        Args:
            structure_name: Name of the structure

        Returns:
            List of nearby structures with spatial relationships
        """
        results = self.client.execute_query(
            cypher_templates.NEARBY_STRUCTURES,
            {"structure_name": structure_name}
        )

        if not results:
            return []

        data = results[0]
        source = data.get("source_structure", structure_name)
        nearby = data.get("nearby_structures", [])

        graph_results = []
        for item in nearby:
            if item.get("structure"):
                relation = item.get("relation", "NEAR")
                graph_results.append(GraphSearchResult(
                    source_entity=source,
                    target_entity=item["structure"],
                    relationship=relation,
                    direction="bidirectional",
                    path_length=1,
                    properties=item.get("properties", {}),
                    context=f"{source} {self._relation_to_turkish(relation)} {item['structure']}"
                ))

        return graph_results

    def get_structure_builders(self, structure_name: str) -> List[GraphSearchResult]:
        """
        Get commissioners, architects, and restorers of a structure.

        Args:
            structure_name: Name of the structure

        Returns:
            List of builder relationships
        """
        results = self.client.execute_query(
            cypher_templates.STRUCTURE_BUILDERS,
            {"structure_name": structure_name}
        )

        if not results:
            return []

        data = results[0]
        structure = data.get("structure", structure_name)

        graph_results = []

        # Add commissioners
        for comm in data.get("commissioners", []):
            if comm.get("id"):
                graph_results.append(GraphSearchResult(
                    source_entity=structure,
                    target_entity=comm["id"],
                    relationship="COMMISSIONED_BY",
                    direction="outgoing",
                    path_length=1,
                    properties={"type": comm.get("type")},
                    context=f"{structure} {comm['id']} tarafından yaptırıldı"
                ))

        # Add builders
        for builder in data.get("builders", []):
            if builder.get("id"):
                graph_results.append(GraphSearchResult(
                    source_entity=structure,
                    target_entity=builder["id"],
                    relationship="BUILT_BY",
                    direction="outgoing",
                    path_length=1,
                    properties={"type": builder.get("type")},
                    context=f"{structure} {builder['id']} tarafından inşa edildi"
                ))

        # Add architects
        for arch in data.get("architects", []):
            if arch.get("id"):
                graph_results.append(GraphSearchResult(
                    source_entity=structure,
                    target_entity=arch["id"],
                    relationship="DESIGNED_BY",
                    direction="outgoing",
                    path_length=1,
                    properties={"type": arch.get("type")},
                    context=f"{structure} {arch['id']} tarafından tasarlandı"
                ))

        # Add restorers
        for rest in data.get("restorers", []):
            if rest.get("id"):
                graph_results.append(GraphSearchResult(
                    source_entity=structure,
                    target_entity=rest["id"],
                    relationship="RESTORED_BY",
                    direction="outgoing",
                    path_length=1,
                    properties={"type": rest.get("type")},
                    context=f"{structure} {rest['id']} tarafından restore edildi"
                ))

        return graph_results

    def multi_hop_search(self,
                         entity_name: str,
                         max_hops: Optional[int] = None,
                         limit: int = 50) -> List[GraphSearchResult]:
        """
        Perform multi-hop graph exploration.

        Args:
            entity_name: Starting entity
            max_hops: Maximum traversal depth
            limit: Maximum results

        Returns:
            List of connected entities with paths
        """
        hops = max_hops or self.max_hops

        # Try APOC version first, fall back to native
        try:
            results = self.client.execute_query(
                cypher_templates.MULTI_HOP_APOC,
                {
                    "entity_name": entity_name,
                    "max_hops": hops,
                    "relationship_filter": "",
                    "limit": limit
                }
            )
        except Exception:
            # APOC not available, use native query
            results = self.client.execute_query(
                cypher_templates.MULTI_HOP_NATIVE,
                {
                    "entity_name": entity_name,
                    "limit": limit
                }
            )

        graph_results = []
        for r in results:
            if r.get("target"):
                path_rels = r.get("path_relations", [])
                graph_results.append(GraphSearchResult(
                    source_entity=r.get("source", entity_name),
                    target_entity=r["target"],
                    relationship=" -> ".join(path_rels) if path_rels else "CONNECTED",
                    direction="outgoing",
                    path_length=r.get("distance", 1),
                    properties={"target_label": r.get("target_label")},
                    context=f"{r.get('source', entity_name)} -> {' -> '.join(path_rels)} -> {r['target']}"
                ))

        return graph_results

    def get_person_students(self, person_name: str) -> List[GraphSearchResult]:
        """
        Find students of a person (e.g., Mimar Sinan's students).

        Args:
            person_name: Name of the teacher/master

        Returns:
            List of student relationships
        """
        results = self.client.execute_query(
            cypher_templates.PERSON_STUDENTS,
            {"person_name": person_name}
        )

        if not results:
            return []

        data = results[0]
        teacher = data.get("teacher", person_name)
        students = data.get("students", [])

        return [
            GraphSearchResult(
                source_entity=teacher,
                target_entity=s["student"],
                relationship="STUDENT_OF",
                direction="incoming",
                path_length=1,
                properties={},
                context=f"{s['student']} {teacher}'in öğrencisidir"
            )
            for s in students if s.get("student")
        ]

    def get_tomb_relationships(self, structure_name: str) -> List[GraphSearchResult]:
        """
        Get tomb and burial relationships for a structure.

        Args:
            structure_name: Name of the structure (e.g., "I. Ahmet Türbesi")

        Returns:
            List of tomb/burial relationships
        """
        results = self.client.execute_query(
            cypher_templates.TOMB_RELATIONSHIPS,
            {"structure_name": structure_name}
        )

        if not results:
            return []

        data = results[0]
        structure = data.get("structure", structure_name)
        graph_results = []

        # Add tomb_of relationships
        for person in data.get("tombs_of", []):
            if person:
                graph_results.append(GraphSearchResult(
                    source_entity=structure,
                    target_entity=person,
                    relationship="TOMB_OF",
                    direction="outgoing",
                    path_length=1,
                    properties={},
                    context=f"{structure} {person}'in türbesidir"
                ))

        # Add burial_place relationships
        for person in data.get("burial_place_for", []):
            if person:
                graph_results.append(GraphSearchResult(
                    source_entity=person,
                    target_entity=structure,
                    relationship="BURIAL_PLACE",
                    direction="outgoing",
                    path_length=1,
                    properties={},
                    context=f"{person} {structure}'de gömülüdür"
                ))

        return graph_results

    def _build_context_string(self,
                              source: str,
                              relation: str,
                              target: str,
                              direction: str) -> str:
        """Build human-readable Turkish context string."""
        relation_tr = self._relation_to_turkish(relation)

        if direction == "outgoing":
            return f"{source} {relation_tr} {target}"
        else:
            return f"{target} {relation_tr} {source}"

    def _relation_to_turkish(self, relation: str) -> str:
        """Convert relationship type to Turkish description."""
        return self.RELATION_DESCRIPTIONS.get(
            relation,
            relation.lower().replace("_", " ")
        )
