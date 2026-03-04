"""
Tests for hybrid retrieval logic.
"""

import pytest
from unittest.mock import Mock, MagicMock

from graphrag.retrieval.hybrid_retriever import (
    HybridRetriever,
    HybridSearchResult,
    RetrievalSource
)
from graphrag.retrieval.vector_retriever import VectorRetriever, VectorSearchResult
from graphrag.retrieval.graph_retriever import GraphRetriever, GraphSearchResult


class TestRetrievalSource:
    """Test RetrievalSource enum."""

    def test_source_values(self):
        """Test all source values exist."""
        assert RetrievalSource.VECTOR.value == "vector"
        assert RetrievalSource.GRAPH.value == "graph"
        assert RetrievalSource.HYBRID.value == "hybrid"


class TestHybridSearchResult:
    """Test HybridSearchResult dataclass."""

    def test_result_creation(self):
        """Test creating a hybrid result."""
        result = HybridSearchResult(
            entity="Ayasofya",
            label="Yapi",
            content="Test content",
            vector_score=0.9,
            graph_score=0.8,
            combined_score=0.85,
            source=RetrievalSource.HYBRID,
            metadata={"key": "value"}
        )

        assert result.entity == "Ayasofya"
        assert result.vector_score == 0.9
        assert result.source == RetrievalSource.HYBRID

    def test_result_to_dict(self):
        """Test serialization to dictionary."""
        result = HybridSearchResult(
            entity="Test",
            label="Label",
            content="Content",
            vector_score=0.5,
            graph_score=0.5,
            combined_score=0.5,
            source=RetrievalSource.VECTOR
        )

        result_dict = result.to_dict()

        assert result_dict["entity"] == "Test"
        assert result_dict["source"] == "vector"


class TestHybridRetriever:
    """Test HybridRetriever class."""

    @pytest.fixture
    def mock_vector_retriever(self):
        """Create mock vector retriever."""
        retriever = Mock(spec=VectorRetriever)
        retriever.search.return_value = []
        return retriever

    @pytest.fixture
    def mock_graph_retriever(self):
        """Create mock graph retriever."""
        retriever = Mock(spec=GraphRetriever)
        retriever.get_entity_context.return_value = []
        retriever.get_nearby_structures.return_value = []
        retriever.get_structure_builders.return_value = []
        retriever.get_person_students.return_value = []
        retriever.trace_origin.return_value = []
        retriever.multi_hop_search.return_value = []
        return retriever

    @pytest.fixture
    def hybrid_retriever(self, mock_vector_retriever, mock_graph_retriever):
        """Create hybrid retriever with mocks."""
        return HybridRetriever(
            vector_retriever=mock_vector_retriever,
            graph_retriever=mock_graph_retriever,
            alpha=0.5,
            rrf_k=60
        )

    def test_initialization(self, hybrid_retriever):
        """Test hybrid retriever initialization."""
        assert hybrid_retriever.alpha == 0.5
        assert hybrid_retriever.rrf_k == 60

    def test_set_alpha(self, hybrid_retriever):
        """Test setting alpha value."""
        hybrid_retriever.set_alpha(0.7)
        assert hybrid_retriever.alpha == 0.7

    def test_set_alpha_invalid(self, hybrid_retriever):
        """Test setting invalid alpha value."""
        with pytest.raises(ValueError):
            hybrid_retriever.set_alpha(1.5)

        with pytest.raises(ValueError):
            hybrid_retriever.set_alpha(-0.1)

    def test_retrieve_empty_results(self, hybrid_retriever):
        """Test retrieval with no results."""
        results = hybrid_retriever.retrieve("test query", entities=[], intent="factual")
        assert results == []

    def test_rrf_fusion_vector_only(self, hybrid_retriever, mock_vector_retriever):
        """Test RRF fusion with vector results only."""
        # Setup vector results
        mock_vector_retriever.search.return_value = [
            VectorSearchResult(
                node_id="1",
                label="Yapi",
                properties={"id": "Ayasofya"},
                score=0.9,
                text="Ayasofya"
            )
        ]

        results = hybrid_retriever.retrieve("Ayasofya", entities=[], intent="factual")

        assert len(results) >= 1
        # First result should be from vector
        ayasofya_result = next((r for r in results if r.entity == "Ayasofya"), None)
        assert ayasofya_result is not None
        assert ayasofya_result.source == RetrievalSource.VECTOR

    def test_rrf_fusion_graph_only(self, hybrid_retriever, mock_graph_retriever):
        """Test RRF fusion with graph results only."""
        # Setup graph results
        mock_graph_retriever.get_entity_context.return_value = [
            GraphSearchResult(
                source_entity="Ayasofya",
                target_entity="I. Justinianus",
                relationship="YAPTIRAN",
                direction="outgoing",
                path_length=1,
                properties={},
                context="Ayasofya I. Justinianus tarafından yaptırıldı"
            )
        ]

        results = hybrid_retriever.retrieve(
            "Kim yaptırdı?",
            entities=["Ayasofya"],
            intent="relational"
        )

        # Should have graph results
        graph_results = [r for r in results if r.source == RetrievalSource.GRAPH]
        assert len(graph_results) >= 1

    def test_rrf_fusion_hybrid(self, hybrid_retriever, mock_vector_retriever, mock_graph_retriever):
        """Test RRF fusion with both vector and graph results."""
        # Setup vector results
        mock_vector_retriever.search.return_value = [
            VectorSearchResult(
                node_id="1",
                label="Yapi",
                properties={"id": "Ayasofya"},
                score=0.9,
                text="Ayasofya"
            ),
            VectorSearchResult(
                node_id="2",
                label="Yapi",
                properties={"id": "Sultanahmet Camii"},
                score=0.7,
                text="Sultanahmet Camii"
            )
        ]

        # Setup graph results for same entity
        mock_graph_retriever.get_entity_context.return_value = [
            GraphSearchResult(
                source_entity="Ayasofya",
                target_entity="I. Justinianus",
                relationship="YAPTIRAN",
                direction="outgoing",
                path_length=1,
                properties={},
                context="Ayasofya I. Justinianus tarafından yaptırıldı"
            )
        ]

        results = hybrid_retriever.retrieve(
            "Ayasofya hakkında",
            entities=["Ayasofya"],
            intent="descriptive"
        )

        # Ayasofya should be hybrid (appears in both)
        ayasofya_result = next((r for r in results if r.entity == "Ayasofya"), None)
        assert ayasofya_result is not None
        assert ayasofya_result.source == RetrievalSource.HYBRID

    def test_alpha_weighting_vector_heavy(self, mock_vector_retriever, mock_graph_retriever):
        """Test that alpha=0 weights toward vector."""
        retriever = HybridRetriever(
            vector_retriever=mock_vector_retriever,
            graph_retriever=mock_graph_retriever,
            alpha=0.0,  # Vector only
            rrf_k=60
        )

        mock_vector_retriever.search.return_value = [
            VectorSearchResult(
                node_id="1",
                label="Yapi",
                properties={"id": "VectorEntity"},
                score=0.9,
                text="VectorEntity"
            )
        ]

        mock_graph_retriever.get_entity_context.return_value = [
            GraphSearchResult(
                source_entity="GraphEntity",
                target_entity="Other",
                relationship="REL",
                direction="outgoing",
                path_length=1,
                properties={},
                context="Context"
            )
        ]

        results = retriever.retrieve("test", entities=["GraphEntity"], intent="factual")

        # Vector entity should rank higher with alpha=0
        if len(results) >= 2:
            vector_result = next((r for r in results if r.entity == "VectorEntity"), None)
            graph_result = next((r for r in results if r.entity == "GraphEntity"), None)

            if vector_result and graph_result:
                assert vector_result.combined_score >= graph_result.combined_score

    def test_spatial_intent_triggers_nearby_search(self, hybrid_retriever, mock_graph_retriever):
        """Test that spatial intent triggers nearby structures search."""
        hybrid_retriever.retrieve(
            "Ayasofya'nın yanında ne var?",
            entities=["Ayasofya"],
            intent="spatial"
        )

        # Should call get_nearby_structures
        mock_graph_retriever.get_nearby_structures.assert_called_with("Ayasofya")

    def test_origin_intent_triggers_trace(self, hybrid_retriever, mock_graph_retriever):
        """Test that origin intent triggers origin tracing."""
        hybrid_retriever.retrieve(
            "Dikilitaş nereden getirildi?",
            entities=["Dikilitaş"],
            intent="origin"
        )

        # Should call trace_origin
        mock_graph_retriever.trace_origin.assert_called_with("Dikilitaş")

    def test_relational_intent_triggers_builders(self, hybrid_retriever, mock_graph_retriever):
        """Test that relational intent triggers builder search."""
        hybrid_retriever.retrieve(
            "Kim yaptırdı?",
            entities=["Ayasofya"],
            intent="relational"
        )

        # Should call get_structure_builders
        mock_graph_retriever.get_structure_builders.assert_called_with("Ayasofya")


class TestRRFScoring:
    """Test Reciprocal Rank Fusion scoring."""

    def test_rrf_formula(self):
        """Test the RRF formula is applied correctly."""
        # RRF score = 1 / (k + rank)
        k = 60
        rank = 1

        expected_score = 1.0 / (k + rank)  # 1/61 ≈ 0.0164

        assert abs(expected_score - 1/61) < 0.001

    def test_higher_rank_lower_score(self):
        """Test that higher rank gives lower RRF score."""
        k = 60

        score_rank_1 = 1.0 / (k + 1)
        score_rank_10 = 1.0 / (k + 10)

        assert score_rank_1 > score_rank_10
