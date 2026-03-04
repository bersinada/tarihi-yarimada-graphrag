#!/usr/bin/env python3
"""
Fix vector index creation for Neo4j.
Diagnoses and creates vector indexes manually.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from graphrag.config import Config
from graphrag.database.neo4j_client import Neo4jClient


def main():
    print("=" * 60)
    print("  Vector Index Fix Script")
    print("=" * 60)

    config = Config.load("config.yaml")
    client = Neo4jClient(
        uri=config.neo4j.uri,
        username=config.neo4j.username,
        password=config.neo4j.password
    )

    try:
        # 1. Check Neo4j version
        print("\n1. Checking Neo4j version...")
        version_query = "CALL dbms.components() YIELD name, versions RETURN name, versions"
        result = client.execute_query(version_query)
        if result:
            for r in result:
                print(f"   {r['name']}: {r['versions']}")

        # 2. Check existing indexes
        print("\n2. Checking existing indexes...")
        index_query = "SHOW INDEXES"
        indexes = client.execute_query(index_query)

        vector_indexes = [idx for idx in indexes if idx.get('type') == 'VECTOR']
        print(f"   Found {len(vector_indexes)} vector indexes")

        for idx in vector_indexes:
            print(f"   - {idx.get('name')}: {idx.get('state')}")

        # 3. Check if nodes have embedding property
        print("\n3. Checking embedding properties on nodes...")
        labels_to_check = ["Document", "Structure", "Building", "Monument", "Person"]

        for label in labels_to_check:
            check_query = f"""
            MATCH (n:{label})
            RETURN count(n) as total,
                   count(n.embedding) as with_embedding,
                   CASE WHEN count(n.embedding) > 0
                        THEN size(head(collect(n.embedding)))
                        ELSE 0 END as embedding_dim
            """
            result = client.execute_query(check_query)
            if result and result[0]['total'] > 0:
                r = result[0]
                print(f"   {label}: {r['with_embedding']}/{r['total']} nodes have embeddings (dim: {r['embedding_dim']})")

        # 4. Try to create vector indexes
        print("\n4. Creating vector indexes...")

        # Labels that should have vector indexes
        index_configs = [
            ("document_embedding", "Document"),
            ("structure_embedding", "Structure"),
            ("building_embedding", "Building"),
            ("monument_embedding", "Monument"),
            ("person_embedding", "Person"),
            ("location_embedding", "Location"),
        ]

        for index_name, label in index_configs:
            # First check if label has nodes with embeddings
            check = client.execute_query(f"""
                MATCH (n:{label}) WHERE n.embedding IS NOT NULL
                RETURN count(n) as cnt
            """)

            if not check or check[0]['cnt'] == 0:
                print(f"   Skipping {index_name} - no {label} nodes with embeddings")
                continue

            # Drop existing index if any
            try:
                client.execute_write(f"DROP INDEX {index_name} IF EXISTS")
            except:
                pass

            # Create new index
            create_query = f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (n:{label})
            ON n.embedding
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """

            try:
                client.execute_write(create_query)
                print(f"   Created: {index_name}")
            except Exception as e:
                print(f"   Failed {index_name}: {e}")

                # Try alternative syntax for older Neo4j
                alt_query = f"""
                CALL db.index.vector.createNodeIndex(
                    '{index_name}',
                    '{label}',
                    'embedding',
                    384,
                    'cosine'
                )
                """
                try:
                    client.execute_write(alt_query)
                    print(f"   Created (alt syntax): {index_name}")
                except Exception as e2:
                    print(f"   Alt syntax also failed: {e2}")

        # 5. Verify indexes
        print("\n5. Verifying indexes...")
        indexes = client.execute_query("SHOW INDEXES WHERE type = 'VECTOR'")

        if indexes:
            print(f"   Success! {len(indexes)} vector indexes created:")
            for idx in indexes:
                print(f"   - {idx.get('name')}: {idx.get('state')}")
        else:
            print("   WARNING: No vector indexes found!")
            print("\n   Your Neo4j version may not support vector indexes.")
            print("   Required: Neo4j 5.11+ or Neo4j AuraDB")
            print("\n   Alternatives:")
            print("   1. Upgrade Neo4j to 5.11+")
            print("   2. Use Neo4j AuraDB (cloud)")
            print("   3. Use external vector store (Pinecone, Weaviate, etc.)")

        # 6. Test vector search
        print("\n6. Testing vector search...")
        if indexes:
            test_query = """
            MATCH (d:Document)
            WHERE d.embedding IS NOT NULL
            WITH d LIMIT 1
            CALL db.index.vector.queryNodes('document_embedding', 5, d.embedding)
            YIELD node, score
            RETURN node.id as id, score
            LIMIT 3
            """
            try:
                results = client.execute_query(test_query)
                if results:
                    print("   Vector search works!")
                    for r in results:
                        print(f"   - {r['id']}: {r['score']:.3f}")
                else:
                    print("   Vector search returned no results")
            except Exception as e:
                print(f"   Vector search failed: {e}")

    finally:
        client.close()

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
