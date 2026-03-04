#!/usr/bin/env python3
"""
One-time setup script for the GraphRAG vector indexes.

This script:
1. Creates vector indexes for all node labels
2. Generates embeddings for existing nodes
3. Processes source documents into Document nodes

Run this once after loading your graph data:
    python scripts/setup_vector_index.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphrag.facade import GraphRAGFacade


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_status(label: str, status: bool):
    """Print status with checkmark or X."""
    icon = "✓" if status else "✗"
    print(f"   {icon} {label}")


def main():
    """Run the full setup process."""
    print_header("Istanbul Tarihi Yarımada GraphRAG Setup")

    print("\nInitializing system...")

    try:
        with GraphRAGFacade() as rag:
            # Check system status first
            print("\n1. Checking system status...")
            status = rag.get_system_status()

            if not status.get("neo4j_connected"):
                print("   ✗ Neo4j connection failed!")
                print(f"   Error: {status.get('error', 'Unknown error')}")
                print("\n   Make sure Neo4j is running:")
                print("   docker-compose up -d")
                return 1

            print("   ✓ Neo4j connected")
            print(f"   Found labels: {', '.join(status.get('node_labels', []))}")

            # Show current node counts
            print("\n   Current node counts:")
            for label, count in status.get("node_counts", {}).items():
                print(f"     - {label}: {count}")

            # Create vector indexes
            print_header("2. Creating Vector Indexes")
            index_results = rag.setup_indexes()

            for label, success in index_results.items():
                print_status(label, success)

            # Count successful indexes
            success_count = sum(1 for s in index_results.values() if s)
            print(f"\n   Created {success_count}/{len(index_results)} indexes")

            # Embed existing nodes
            print_header("3. Generating Node Embeddings")
            print("   This may take a few minutes...")

            embed_results = rag.embed_all_nodes(force=False)

            total_embedded = 0
            for label, count in embed_results.items():
                if count > 0:
                    print(f"   ✓ {label}: {count} nodes embedded")
                    total_embedded += count
                elif count == 0:
                    print(f"   - {label}: already embedded or no nodes")
                else:
                    print(f"   ✗ {label}: embedding failed")

            print(f"\n   Total: {total_embedded} nodes embedded")

            # Process documents
            print_header("4. Processing Source Documents")

            # Check if source directory exists
            source_dir = Path(rag.config.documents.source_dir)
            if not source_dir.exists():
                print(f"   ✗ Source directory not found: {source_dir}")
                print("   Skipping document processing")
            else:
                print(f"   Source directory: {source_dir}")
                doc_results = rag.process_documents(clear_existing=True)

                total_chunks = 0
                for filename, count in doc_results.items():
                    if count > 0:
                        print(f"   ✓ {filename}: {count} chunks")
                        total_chunks += count
                    else:
                        print(f"   ✗ {filename}: processing failed")

                print(f"\n   Total: {total_chunks} document chunks created")

            # Final status
            print_header("5. Setup Complete!")

            final_status = rag.get_system_status()
            print("\n   Final statistics:")
            print(f"   - Vector indexes: {len(final_status.get('vector_indexes', []))}")

            doc_stats = final_status.get("document_stats", {})
            if doc_stats:
                print(f"   - Document nodes: {doc_stats.get('total', 0)}")
                print(f"   - Linked structures: {doc_stats.get('linked_structures', 0)}")

            print("\n   You can now use the system:")
            print("   python scripts/query_cli.py")
            print("\n   Or in Python:")
            print("   from graphrag import ask")
            print("   answer = ask('Ayasofya hakkında bilgi ver')")

            return 0

    except FileNotFoundError as e:
        print(f"\n   ✗ Configuration error: {e}")
        print("\n   Make sure you have:")
        print("   1. Created config.yaml (copy from template)")
        print("   2. Created .env with your GROQ_API_KEY")
        return 1

    except Exception as e:
        print(f"\n   ✗ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
