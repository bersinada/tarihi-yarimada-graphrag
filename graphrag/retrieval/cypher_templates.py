"""
Pre-built Cypher query templates for graph traversal.

These templates are used by the GraphRetriever to execute
various types of graph queries based on query intent.

Updated for English labels in Neo4j:
- Structure, Building, Monument (yapılar)
- Person (kişiler - padişahlar, mimarlar, imparatorlar)
- Location, City, Region, Street (konumlar)
- Event (olaylar)
- Organization, Group (organizasyonlar)
- Deity (tanrılar)
- Document (RAG dökümanları)
"""

# ==============================================================================
# ENTITY CONTEXT QUERIES
# ==============================================================================

# Get all relationships for a specific entity
ENTITY_FULL_CONTEXT = """
MATCH (n)
WHERE (n.id = $entity_name
   OR toLower(n.id) = toLower($entity_name)
   OR toLower(n.id) CONTAINS toLower($entity_name)
   OR n.name = $entity_name)
  AND NOT n.id CONTAINS '_chunk_'
  AND n.chunk_index IS NULL
OPTIONAL MATCH (n)-[r]-(related)
WHERE NOT related.id CONTAINS '_chunk_'
RETURN n as entity,
       labels(n)[0] as entity_label,
       properties(n) as entity_props,
       collect(DISTINCT {
           related_id: related.id,
           related_label: labels(related)[0],
           relation: type(r),
           direction: CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END,
           rel_props: properties(r)
       }) as relationships
ORDER BY CASE WHEN toLower(n.id) = toLower($entity_name) THEN 0 ELSE 1 END
LIMIT 1
"""

# Get entity with all its properties
ENTITY_PROPERTIES = """
MATCH (n)
WHERE (n.id = $entity_name
   OR toLower(n.id) = toLower($entity_name)
   OR toLower(n.id) CONTAINS toLower($entity_name))
  AND NOT n.id CONTAINS '_chunk_'
  AND n.chunk_index IS NULL
RETURN n, labels(n)[0] as label, properties(n) as props
ORDER BY CASE WHEN toLower(n.id) = toLower($entity_name) THEN 0 ELSE 1 END
LIMIT 1
"""

# ==============================================================================
# MULTI-HOP TRAVERSAL QUERIES
# ==============================================================================

# Multi-hop path exploration using APOC (if available)
MULTI_HOP_APOC = """
MATCH (start)
WHERE start.id = $entity_name OR toLower(start.id) CONTAINS toLower($entity_name)
CALL apoc.path.subgraphNodes(start, {
    maxLevel: $max_hops,
    relationshipFilter: $relationship_filter
}) YIELD node
WITH start, node
WHERE node <> start
OPTIONAL MATCH path = shortestPath((start)-[*1..3]-(node))
RETURN start.id as source,
       node.id as target,
       labels(node)[0] as target_label,
       [r IN relationships(path) | type(r)] as path_relations,
       length(path) as distance
ORDER BY distance
LIMIT $limit
"""

# Multi-hop without APOC (fallback)
MULTI_HOP_NATIVE = """
MATCH (start)
WHERE (start.id = $entity_name OR toLower(start.id) CONTAINS toLower($entity_name))
  AND NOT start.id CONTAINS '_chunk_'
  AND start.chunk_index IS NULL
MATCH path = (start)-[*1..3]-(end)
WHERE start <> end
  AND NOT end.id CONTAINS '_chunk_'
WITH start, end, path, length(path) as distance
RETURN DISTINCT
    start.id as source,
    end.id as target,
    labels(end)[0] as target_label,
    [r IN relationships(path) | type(r)] as path_relations,
    distance
ORDER BY distance
LIMIT $limit
"""

# ==============================================================================
# ORIGIN TRACING QUERIES (for Dikilitaş-type queries)
# ==============================================================================

# Trace origin/journey of an artifact
TRACE_ORIGIN = """
MATCH (artifact)
WHERE (artifact.id = $artifact_name
   OR toLower(artifact.id) CONTAINS toLower($artifact_name))
  AND NOT artifact.id CONTAINS '_chunk_'
  AND artifact.chunk_index IS NULL
OPTIONAL MATCH (artifact)-[:ORIGINALLY_LOCATED_IN]->(origin_place)
OPTIONAL MATCH (artifact)-[:TRANSFERRED_TO]->(transfer_place)
OPTIONAL MATCH (artifact)-[:ORIGINATED_FROM]->(origin_region)
OPTIONAL MATCH (artifact)-[:COMMISSIONED_BY]->(commissioner)
OPTIONAL MATCH (artifact)-[:BUILT_BY]->(builder)
RETURN artifact.id as artifact,
       labels(artifact)[0] as artifact_type,
       properties(artifact) as artifact_props,
       origin_place.id as original_location,
       origin_region.id as origin_region,
       transfer_place.id as transferred_to,
       commissioner.id as commissioned_by,
       labels(commissioner)[0] as commissioner_type,
       builder.id as built_by,
       labels(builder)[0] as builder_type
"""

# ==============================================================================
# SPATIAL RELATIONSHIP QUERIES
# ==============================================================================

# Find nearby structures
NEARBY_STRUCTURES = """
MATCH (s)
WHERE (s:Structure OR s:Building OR s:Monument)
  AND (s.id = $structure_name OR toLower(s.id) CONTAINS toLower($structure_name))
OPTIONAL MATCH (s)-[r:NEAR]-(nearby)
WHERE nearby:Structure OR nearby:Building OR nearby:Monument
RETURN s.id as source_structure,
       collect(DISTINCT {
           structure: nearby.id,
           relation: type(r),
           label: labels(nearby)[0],
           properties: properties(nearby)
       }) as nearby_structures
"""

# Structures in a specific location
STRUCTURES_IN_LOCATION = """
MATCH (s)-[:LOCATED_IN]->(loc)
WHERE (s:Structure OR s:Building OR s:Monument)
  AND (loc.id = $location_name OR toLower(loc.id) CONTAINS toLower($location_name))
RETURN loc.id as location,
       labels(loc)[0] as location_type,
       collect({
           structure: s.id,
           label: labels(s)[0],
           properties: properties(s)
       }) as structures
"""

# All spatial relationships for a structure
SPATIAL_CONTEXT = """
MATCH (s)
WHERE (s:Structure OR s:Building OR s:Monument)
  AND (s.id = $structure_name OR toLower(s.id) CONTAINS toLower($structure_name))
OPTIONAL MATCH (s)-[:LOCATED_IN]->(loc)
OPTIONAL MATCH (s)-[spatial:NEAR]-(nearby)
WHERE nearby:Structure OR nearby:Building OR nearby:Monument
RETURN s.id as structure,
       loc.id as location,
       labels(loc)[0] as location_type,
       collect(DISTINCT {
           nearby: nearby.id,
           nearby_label: labels(nearby)[0],
           relation: type(spatial)
       }) as spatial_relations
"""

# ==============================================================================
# RELATIONAL QUERIES (Builder, Architect, etc.)
# ==============================================================================

# Find structures by builder/commissioner
STRUCTURES_BY_BUILDER = """
MATCH (s)-[:COMMISSIONED_BY|BUILT_BY]->(person:Person)
WHERE (s:Structure OR s:Building OR s:Monument)
  AND (person.id = $person_name OR toLower(person.id) CONTAINS toLower($person_name))
RETURN person.id as builder,
       collect({
           structure: s.id,
           label: labels(s)[0],
           properties: properties(s)
       }) as structures
"""

# Find structures by architect/designer
STRUCTURES_BY_ARCHITECT = """
MATCH (s)-[:DESIGNED_BY]->(architect:Person)
WHERE (s:Structure OR s:Building OR s:Monument)
  AND (architect.id = $architect_name OR toLower(architect.id) CONTAINS toLower($architect_name))
RETURN architect.id as architect,
       collect({
           structure: s.id,
           label: labels(s)[0],
           properties: properties(s)
       }) as structures
"""

# Find person's students (for Mimar Sinan queries)
PERSON_STUDENTS = """
MATCH (student:Person)-[:STUDENT_OF]->(teacher:Person)
WHERE teacher.id = $person_name
   OR toLower(teacher.id) CONTAINS toLower($person_name)
RETURN teacher.id as teacher,
       collect({
           student: student.id
       }) as students
"""

# Find who commissioned/built a structure
STRUCTURE_BUILDERS = """
MATCH (s)
WHERE (s:Structure OR s:Building OR s:Monument)
  AND (s.id = $structure_name OR toLower(s.id) CONTAINS toLower($structure_name))
OPTIONAL MATCH (s)-[:COMMISSIONED_BY]->(commissioner)
OPTIONAL MATCH (s)-[:BUILT_BY]->(builder)
OPTIONAL MATCH (s)-[:DESIGNED_BY]->(architect)
OPTIONAL MATCH (s)-[:RESTORED_BY]->(restorer)
RETURN s.id as structure,
       properties(s) as structure_props,
       collect(DISTINCT {id: commissioner.id, type: labels(commissioner)[0], role: 'commissioner'}) as commissioners,
       collect(DISTINCT {id: builder.id, type: labels(builder)[0], role: 'builder'}) as builders,
       collect(DISTINCT {id: architect.id, type: labels(architect)[0], role: 'architect'}) as architects,
       collect(DISTINCT {id: restorer.id, type: labels(restorer)[0], role: 'restorer'}) as restorers
"""

# ==============================================================================
# TOMB/BURIAL QUERIES
# ==============================================================================

# Find tombs and burial places
TOMB_RELATIONSHIPS = """
MATCH (s)
WHERE (s:Structure OR s:Building)
  AND (s.id = $structure_name OR toLower(s.id) CONTAINS toLower($structure_name))
OPTIONAL MATCH (s)-[:TOMB_OF]->(buried_person:Person)
OPTIONAL MATCH (person:Person)-[:BURIAL_PLACE]->(s)
RETURN s.id as structure,
       collect(DISTINCT buried_person.id) as tombs_of,
       collect(DISTINCT person.id) as burial_place_for
"""

# Find where a person is buried
PERSON_BURIAL = """
MATCH (p:Person)
WHERE p.id = $person_name OR toLower(p.id) CONTAINS toLower($person_name)
OPTIONAL MATCH (p)-[:BURIAL_PLACE]->(burial_place)
OPTIONAL MATCH (tomb)-[:TOMB_OF]->(p)
RETURN p.id as person,
       burial_place.id as burial_place,
       tomb.id as tomb
"""

# ==============================================================================
# EVENT AND HISTORICAL QUERIES
# ==============================================================================

# Events related to a structure
STRUCTURE_EVENTS = """
MATCH (s)
WHERE (s:Structure OR s:Building OR s:Monument)
  AND (s.id = $structure_name OR toLower(s.id) CONTAINS toLower($structure_name))
OPTIONAL MATCH (s)-[r]-(e:Event)
RETURN s.id as structure,
       collect({
           event: e.id,
           relation: type(r),
           properties: properties(e)
       }) as events
"""

# Monument commemorations
MONUMENT_COMMEMORATES = """
MATCH (m:Monument)-[:COMMEMORATES]->(event:Event)
WHERE m.id = $monument_name OR toLower(m.id) CONTAINS toLower($monument_name)
RETURN m.id as monument,
       collect({
           event: event.id,
           properties: properties(event)
       }) as commemorates
"""

# ==============================================================================
# COMPARISON QUERIES
# ==============================================================================

# Compare two structures
COMPARE_STRUCTURES = """
MATCH (s1), (s2)
WHERE (s1:Structure OR s1:Building OR s1:Monument)
  AND (s2:Structure OR s2:Building OR s2:Monument)
  AND (s1.id = $structure1 OR toLower(s1.id) = toLower($structure1) OR toLower(s1.id) CONTAINS toLower($structure1))
  AND (s2.id = $structure2 OR toLower(s2.id) = toLower($structure2) OR toLower(s2.id) CONTAINS toLower($structure2))
OPTIONAL MATCH (s1)-[r1]->(common)<-[r2]-(s2)
RETURN s1.id as structure1,
       properties(s1) as props1,
       labels(s1)[0] as label1,
       s2.id as structure2,
       properties(s2) as props2,
       labels(s2)[0] as label2,
       collect(DISTINCT {
           common: common.id,
           common_type: labels(common)[0],
           rel1: type(r1),
           rel2: type(r2)
       }) as common_connections
"""

# ==============================================================================
# DOCUMENT QUERIES
# ==============================================================================

# Get documents related to a structure
DOCUMENTS_FOR_STRUCTURE = """
MATCH (d:Document)-[:DESCRIBES]->(s)
WHERE (s:Structure OR s:Building OR s:Monument)
  AND (s.id = $structure_name OR toLower(s.id) CONTAINS toLower($structure_name))
RETURN s.id as structure,
       collect({
           doc_id: d.id,
           content: d.content,
           source: d.source_file,
           chunk_index: d.chunk_index
       }) as documents
ORDER BY d.chunk_index
"""

# ==============================================================================
# DEITY AND DEDICATION QUERIES
# ==============================================================================

# Find dedications
DEDICATIONS = """
MATCH (s)-[:DEDICATED_TO]->(d:Deity)
WHERE (s:Structure OR s:Building OR s:Monument)
  AND (s.id = $structure_name OR toLower(s.id) CONTAINS toLower($structure_name))
RETURN s.id as structure,
       collect({
           deity: d.id,
           properties: properties(d)
       }) as dedicated_to
"""

# ==============================================================================
# CONVERSION QUERIES (e.g., church to mosque)
# ==============================================================================

# Find conversion history
CONVERSION_HISTORY = """
MATCH (s)
WHERE (s:Structure OR s:Building)
  AND (s.id = $structure_name OR toLower(s.id) CONTAINS toLower($structure_name))
OPTIONAL MATCH (s)-[:CONVERTED_TO_MOSQUE_BY]->(converter:Person)
RETURN s.id as structure,
       converter.id as converted_by,
       properties(s) as structure_props
"""

# ==============================================================================
# FULL STRUCTURE CONTEXT (combines multiple queries)
# ==============================================================================

FULL_STRUCTURE_CONTEXT = """
MATCH (s)
WHERE (s:Structure OR s:Building OR s:Monument)
  AND (s.id = $structure_name OR toLower(s.id) CONTAINS toLower($structure_name))

// Get basic info
WITH s

// Get all relationships
OPTIONAL MATCH (s)-[r]-(related)

RETURN s.id as structure,
       labels(s) as labels,
       properties(s) as properties,
       collect(DISTINCT {
           related_id: related.id,
           related_label: labels(related)[0],
           relation: type(r),
           direction: CASE WHEN startNode(r) = s THEN 'outgoing' ELSE 'incoming' END
       }) as all_relationships
"""
