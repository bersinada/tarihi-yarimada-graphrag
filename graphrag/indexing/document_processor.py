"""
Document processing for creating and embedding document chunks.

Reads source text files, chunks them, creates Document nodes,
and links them to relevant Structure/Building/Monument nodes.

Updated for English labels in Neo4j schema.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..database.neo4j_client import Neo4jClient
from ..embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processes source documents for the GraphRAG system.

    Responsibilities:
    - Read text files from source directory
    - Chunk documents into manageable segments
    - Create Document nodes in Neo4j
    - Link documents to related Structure/Building/Monument nodes
    - Generate and store embeddings

    Example:
        >>> processor = DocumentProcessor(client, embedder, "son-veri")
        >>> processor.process_all_documents()
    """

    # Mapping of file names (normalized) to Neo4j node id values
    # Based on actual files in son-veri/ and Neo4j node ids
    FILE_TO_STRUCTURE = {
        # Exact matches for son-veri/*.txt files
        "ayasofya": "Ayasofya",
        "ayairini": "Aya İrini",
        "dikilitas": "Dikilitaş",
        "yilanlisutun": "Yılanlı Sütun",
        "ormedikilitas": "Örme Dikilitaş",
        "almancesmesi": "Alman Çeşmesi",
        "iii.ahmetcesmesi": "III. Ahmet Çeşmesi",
        "sultanahmetcamii": "Sultanahmet Camii",
        "firuzaga": "Firuz Ağa Camii",
        "i.ahmetturbesi": "I. Ahmet Türbesi",
        "İ.ahmetturbesi": "I. Ahmet Türbesi",  # Turkish İ variant

        # Alternative spellings
        "aya_irini": "Aya İrini",
        "yilanli_sutun": "Yılanlı Sütun",
        "orme_dikilitas": "Örme Dikilitaş",
        "alman_cesmesi": "Alman Çeşmesi",
        "ahmed_cesmesi": "III. Ahmet Çeşmesi",
        "sultanahmet": "Sultanahmet Camii",
        "firuz_aga": "Firuz Ağa Camii",
        "ahmet_turbesi": "I. Ahmet Türbesi",
    }

    def __init__(self,
                 client: Neo4jClient,
                 embedder: BaseEmbedder,
                 source_dir: str = "son-veri",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        """
        Initialize the document processor.

        Args:
            client: Neo4j client instance
            embedder: Embedding provider
            source_dir: Directory containing source text files
            chunk_size: Target size for text chunks (in characters)
            chunk_overlap: Overlap between consecutive chunks
        """
        self.client = client
        self.embedder = embedder
        self.source_dir = Path(source_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_all_documents(self) -> Dict[str, int]:
        """
        Process all text files in the source directory.

        Returns:
            Dictionary mapping filename to number of chunks created
        """
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")

        results = {}
        txt_files = list(self.source_dir.glob("*.txt"))

        logger.info(f"Found {len(txt_files)} text files in {self.source_dir}")

        for file_path in txt_files:
            try:
                count = self.process_document(file_path)
                results[file_path.name] = count
                logger.info(f"Processed {file_path.name}: {count} chunks")
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                results[file_path.name] = -1

        return results

    def process_document(self, file_path: Path) -> int:
        """
        Process a single document file.

        Args:
            file_path: Path to the text file

        Returns:
            Number of chunks created
        """
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract structure name from filename
        structure_name = self._get_structure_name(file_path.stem)

        if structure_name:
            logger.info(f"Mapped {file_path.stem} -> {structure_name}")
        else:
            logger.warning(f"No structure mapping found for: {file_path.stem}")

        # Chunk the content
        chunks = self._chunk_text(content)

        # Create Document nodes with embeddings
        chunk_count = 0
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{file_path.stem}_chunk_{i}"

            # Create document node
            self._create_document_node(
                chunk_id=chunk_id,
                content=chunk_text,
                source_file=file_path.name,
                chunk_index=i,
                total_chunks=len(chunks),
                structure_name=structure_name
            )
            chunk_count += 1

        return chunk_count

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        Uses paragraph boundaries when possible, falls back to
        character-based chunking.

        Args:
            text: Full text content

        Returns:
            List of text chunks
        """
        # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        if len(text) <= self.chunk_size:
            return [text]

        chunks = []

        # Split by paragraphs first
        paragraphs = text.split('\n\n')

        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) + 2 > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # If paragraph itself is too long, split it
                if len(para) > self.chunk_size:
                    sub_chunks = self._split_long_text(para)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1]
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # Add remaining content
        if current_chunk:
            chunks.append(current_chunk.strip())

        # Add overlap between chunks
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)

        return chunks

    def _split_long_text(self, text: str) -> List[str]:
        """
        Split text that exceeds chunk size.

        Args:
            text: Text to split

        Returns:
            List of smaller chunks
        """
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 > self.chunk_size:
                if current:
                    chunks.append(current.strip())
                current = sentence
            else:
                if current:
                    current += " " + sentence
                else:
                    current = sentence

        if current:
            chunks.append(current.strip())

        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """
        Add overlap from previous chunk to each chunk.

        Args:
            chunks: List of non-overlapping chunks

        Returns:
            List of overlapping chunks
        """
        overlapped = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            # Get last N characters of previous chunk
            overlap_text = prev_chunk[-self.chunk_overlap:]

            # Try to break at word boundary
            space_idx = overlap_text.find(' ')
            if space_idx > 0:
                overlap_text = overlap_text[space_idx + 1:]

            overlapped.append(overlap_text + " " + chunks[i])

        return overlapped

    def _get_structure_name(self, filename_stem: str) -> Optional[str]:
        """
        Map filename to structure name for linking.

        Args:
            filename_stem: Filename without extension

        Returns:
            Structure name or None
        """
        # Try exact match first
        if filename_stem in self.FILE_TO_STRUCTURE:
            return self.FILE_TO_STRUCTURE[filename_stem]

        # Normalize filename (lowercase, remove special chars)
        normalized = filename_stem.lower().replace("-", "").replace("_", "").replace(" ", "")

        # Check normalized mapping
        for key, value in self.FILE_TO_STRUCTURE.items():
            norm_key = key.lower().replace("-", "").replace("_", "").replace(" ", "").replace(".", "")
            if norm_key == normalized or normalized in norm_key or norm_key in normalized:
                return value

        return None

    def _create_document_node(self,
                              chunk_id: str,
                              content: str,
                              source_file: str,
                              chunk_index: int,
                              total_chunks: int,
                              structure_name: Optional[str]) -> None:
        """
        Create a Document node in Neo4j with embedding.

        Args:
            chunk_id: Unique identifier for the chunk
            content: Text content of the chunk
            source_file: Source filename
            chunk_index: Index of chunk in document
            total_chunks: Total number of chunks
            structure_name: Related structure name for linking
        """
        # Generate embedding
        embedding = self.embedder.embed_text(content)

        # Create Document node
        create_query = """
        MERGE (d:Document {id: $chunk_id})
        SET d.content = $content,
            d.source_file = $source_file,
            d.chunk_index = $chunk_index,
            d.total_chunks = $total_chunks,
            d.embedding = $embedding,
            d.char_count = $char_count
        """

        self.client.execute_write(create_query, {
            "chunk_id": chunk_id,
            "content": content,
            "source_file": source_file,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "embedding": embedding,
            "char_count": len(content)
        })

        # Link to Structure/Building/Monument node if structure name is known
        if structure_name:
            link_query = """
            MATCH (d:Document {id: $chunk_id})
            MATCH (s)
            WHERE (s:Structure OR s:Building OR s:Monument)
              AND (s.id = $structure_name OR toLower(s.id) = toLower($structure_name))
            MERGE (d)-[:DESCRIBES]->(s)
            """

            self.client.execute_write(link_query, {
                "chunk_id": chunk_id,
                "structure_name": structure_name
            })

    def clear_documents(self) -> int:
        """
        Remove all Document nodes and their relationships.

        Returns:
            Number of nodes deleted
        """
        result = self.client.execute_write("""
            MATCH (d:Document)
            DETACH DELETE d
        """)
        count = result.get("nodes_deleted", 0) if result else 0
        logger.info(f"Deleted {count} Document nodes")
        return count

    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get statistics about Document nodes.

        Returns:
            Dictionary with document statistics
        """
        stats_query = """
        MATCH (d:Document)
        WITH count(d) as total,
             sum(d.char_count) as total_chars,
             avg(d.char_count) as avg_chars
        OPTIONAL MATCH (d:Document)-[:DESCRIBES]->(s)
        WHERE s:Structure OR s:Building OR s:Monument
        WITH total, total_chars, avg_chars, count(DISTINCT s) as linked_structures
        RETURN total, total_chars, avg_chars, linked_structures
        """

        result = self.client.execute_query(stats_query)

        if result:
            return result[0]
        return {"total": 0, "total_chars": 0, "avg_chars": 0, "linked_structures": 0}
