"""
GraphRAG Facade - Kompakt ana giris noktasi.
3 asamali pipeline: Analiz -> Getir -> Cevapla
Mekansal sorgularda mesafe hesaplama dahil.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from .config import Config
from .database.neo4j_client import Neo4jClient
from .embeddings import get_embedder
from .indexing.vector_index import VectorIndexManager
from .indexing.document_processor import DocumentProcessor
from .retrieval.vector_retriever import VectorRetriever
from .retrieval.graph_retriever import GraphRetriever
from .retrieval.hybrid_retriever import HybridRetriever
from .query.analyzer import QueryAnalyzer, QueryAnalysis
from .generation.response_generator import ResponseGenerator
from .utils.spatial_utils import haversine, cardinal_direction_tr, format_distance

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Sorgu sonucu."""
    query: str
    analysis: QueryAnalysis
    response: str
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "analysis": self.analysis.to_dict(),
            "response": self.response,
            "sources": self.sources,
            "metadata": self.metadata,
        }


class GraphRAGFacade:
    """Istanbul Tarihi Yarimada GraphRAG sistemi."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self._setup_logging()
        logger.info("Initializing GraphRAG system...")

        # Neo4j
        self.client = Neo4jClient(
            uri=self.config.neo4j.uri,
            username=self.config.neo4j.username,
            password=self.config.neo4j.password,
        )

        # LLM
        self.llm = self._init_llm()

        # Embedder
        self.embedder = get_embedder(
            provider=self.config.embeddings.provider,
            model=self.config.embeddings.model,
        )

        # Retriever'lar
        self.vector_retriever = VectorRetriever(
            client=self.client,
            embedder=self.embedder,
            top_k=self.config.retrieval.vector_top_k,
            min_score=self.config.retrieval.min_similarity,
        )
        self.graph_retriever = GraphRetriever(
            client=self.client,
            max_hops=self.config.retrieval.graph_max_hops,
        )
        self.hybrid_retriever = HybridRetriever(
            vector_retriever=self.vector_retriever,
            graph_retriever=self.graph_retriever,
            alpha=self.config.retrieval.hybrid_alpha,
            rrf_k=self.config.retrieval.rrf_k,
        )

        # Analyzer & Generator
        self.query_analyzer = QueryAnalyzer(
            llm=self.llm,
            fallback_to_rules=self.config.query.fallback_to_rules,
        )
        self.response_generator = ResponseGenerator(llm=self.llm)

        # Setup araclari
        self.index_manager = VectorIndexManager(
            client=self.client,
            embedder=self.embedder,
            dimension=self.config.embeddings.dimension,
        )
        self.document_processor = DocumentProcessor(
            client=self.client,
            embedder=self.embedder,
            source_dir=self.config.documents.source_dir,
            chunk_size=self.config.documents.chunk_size,
            chunk_overlap=self.config.documents.chunk_overlap,
        )

        logger.info("GraphRAG system initialized successfully")

    # ------------------------------------------------------------------
    # Ana sorgu pipeline
    # ------------------------------------------------------------------

    def query(self, user_query: str) -> QueryResult:
        """3 asamali pipeline: Analiz -> Getir -> Cevapla."""
        logger.info(f"Processing query: {user_query}")

        # 1. Analiz
        analysis = self.query_analyzer.analyze(user_query)
        logger.info(f"Intent: {analysis.intent.value}, Entities: {analysis.entities}")

        # 2. Mekansal sorgu ise mesafe hesapla
        spatial_summary = ""
        if analysis.intent.value == "spatial" and len(analysis.entities) >= 2:
            spatial_summary = self._compute_distance(
                analysis.entities[0], analysis.entities[1]
            )

        # 3. Hybrid retrieval
        results = self.hybrid_retriever.retrieve(
            query=user_query,
            entities=analysis.entities,
            intent=analysis.intent.value,
        )
        logger.info(f"Retrieval: {len(results)} results")

        # 4. LLM ile cevap uret
        response = self.response_generator.generate(
            user_query, results,
            include_sources=False,
            spatial_summary=spatial_summary,
        )

        return QueryResult(
            query=user_query,
            analysis=analysis,
            response=response,
            sources=[r.entity for r in results[:5] if r.entity],
            metadata={
                "intent": analysis.intent.value,
                "confidence": analysis.confidence,
                "result_count": len(results),
                "spatial_computed": bool(spatial_summary),
            },
        )

    # ------------------------------------------------------------------
    # Mesafe hesaplama
    # ------------------------------------------------------------------

    def _compute_distance(self, entity_a: str, entity_b: str) -> str:
        """Iki yapi arasindaki mesafeyi hesapla."""
        props_a = self._get_entity_coords(entity_a)
        props_b = self._get_entity_coords(entity_b)

        if not props_a or not props_b:
            missing = []
            if not props_a:
                missing.append(entity_a)
            if not props_b:
                missing.append(entity_b)
            logger.warning(f"Koordinat bulunamadi: {', '.join(missing)}")
            return ""

        lat_a, lon_a = props_a
        lat_b, lon_b = props_b

        dist_m = haversine(lat_a, lon_a, lat_b, lon_b)
        direction = cardinal_direction_tr(lat_a, lon_a, lat_b, lon_b)
        dist_str = format_distance(dist_m)

        return (
            f"**{entity_a}** ile **{entity_b}** arasindaki kus ucusu mesafe "
            f"**{dist_str}**'dir. "
            f"{entity_b}, {entity_a}'nin {direction} yonundedir.\n"
            f"Koordinatlar: {entity_a}: {lat_a:.5f}K, {lon_a:.5f}D | "
            f"{entity_b}: {lat_b:.5f}K, {lon_b:.5f}D"
        )

    def _get_entity_coords(self, entity_name: str):
        """Neo4j'den yapinin koordinatlarini getir."""
        results = self.client.execute_query(
            """
            MATCH (n)
            WHERE (n.id = $name OR toLower(n.id) = toLower($name)
                   OR toLower(n.id) CONTAINS toLower($name))
              AND NOT n.id CONTAINS '_chunk_'
              AND n.chunk_index IS NULL
              AND n.latitude IS NOT NULL AND n.longitude IS NOT NULL
            RETURN toFloat(n.latitude) AS lat, toFloat(n.longitude) AS lon
            ORDER BY CASE WHEN toLower(n.id) = toLower($name) THEN 0 ELSE 1 END
            LIMIT 1
            """,
            {"name": entity_name},
        )
        if results:
            return (results[0]["lat"], results[0]["lon"])
        return None

    # ------------------------------------------------------------------
    # Setup & sistem
    # ------------------------------------------------------------------

    def setup_indexes(self, labels=None):
        return self.index_manager.create_indexes(labels)

    def embed_all_nodes(self, force=False):
        return self.index_manager.embed_all_labels(force=force)

    def process_documents(self, clear_existing=False):
        if clear_existing:
            self.document_processor.clear_documents()
        return self.document_processor.process_all_documents()

    def get_system_status(self) -> Dict[str, Any]:
        status = {
            "neo4j_connected": False,
            "vector_indexes": [],
            "node_counts": {},
            "document_stats": {},
        }
        try:
            labels = self.client.get_node_labels()
            status["neo4j_connected"] = True
            for label in labels:
                status["node_counts"][label] = self.client.get_node_count(label)
            status["vector_indexes"] = self.index_manager.get_index_status()
            status["document_stats"] = self.document_processor.get_document_stats()
        except Exception as e:
            status["error"] = str(e)
        return status

    def set_retrieval_alpha(self, alpha: float):
        self.hybrid_retriever.set_alpha(alpha)

    # ------------------------------------------------------------------
    # Config & lifecycle
    # ------------------------------------------------------------------

    def _load_config(self, config_path: str) -> Config:
        from dotenv import load_dotenv
        load_dotenv()
        return Config.load(config_path)

    def _setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level),
            format=self.config.logging.format,
        )

    def _init_llm(self) -> ChatGoogleGenerativeAI:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        return ChatGoogleGenerativeAI(
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            google_api_key=api_key,
        )

    def close(self):
        if self.client:
            self.client.close()
        logger.info("GraphRAG system closed")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
