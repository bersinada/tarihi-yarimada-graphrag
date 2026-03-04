"""
Tests for query analysis module.
"""

import pytest
from graphrag.query.analyzer import QueryAnalyzer, QueryIntent, QueryAnalysis


class TestQueryIntent:
    """Test QueryIntent enum."""

    def test_intent_values(self):
        """Test all intent values exist."""
        assert QueryIntent.FACTUAL.value == "factual"
        assert QueryIntent.RELATIONAL.value == "relational"
        assert QueryIntent.SPATIAL.value == "spatial"
        assert QueryIntent.ORIGIN.value == "origin"
        assert QueryIntent.TEMPORAL.value == "temporal"
        assert QueryIntent.COMPARATIVE.value == "comparative"
        assert QueryIntent.DESCRIPTIVE.value == "descriptive"


class TestQueryAnalyzerRuleBased:
    """Test rule-based query analysis (no LLM)."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer without LLM."""
        return QueryAnalyzer(llm=None, fallback_to_rules=True)

    def test_spatial_intent_yaninda(self, analyzer):
        """Test detection of spatial queries with 'yanında'."""
        result = analyzer.analyze("Ayasofya'nın yanında ne var?")
        assert result.intent == QueryIntent.SPATIAL
        assert "Ayasofya" in result.entities

    def test_spatial_intent_karsisinda(self, analyzer):
        """Test detection of spatial queries with 'karşısında'."""
        result = analyzer.analyze("Sultanahmet Camii'nin karşısında ne var?")
        assert result.intent == QueryIntent.SPATIAL
        assert "Sultanahmet Camii" in result.entities

    def test_origin_intent_nereden(self, analyzer):
        """Test detection of origin queries with 'nereden'."""
        result = analyzer.analyze("Dikilitaş nereden getirildi?")
        assert result.intent == QueryIntent.ORIGIN
        assert "Dikilitaş" in result.entities

    def test_origin_intent_getirtti(self, analyzer):
        """Test detection of origin queries with 'getirtti'."""
        result = analyzer.analyze("Dikilitaşı Mısır'dan kim getirtti?")
        assert result.intent == QueryIntent.ORIGIN
        assert "Dikilitaş" in result.entities

    def test_relational_intent_kim_yaptirdi(self, analyzer):
        """Test detection of relational queries."""
        result = analyzer.analyze("Sultanahmet Camii'ni kim yaptırdı?")
        assert result.intent == QueryIntent.RELATIONAL
        assert "Sultanahmet Camii" in result.entities

    def test_relational_intent_ogrenci(self, analyzer):
        """Test detection of student relationship queries."""
        result = analyzer.analyze("Mimar Sinan'ın öğrencileri kimler?")
        assert result.intent == QueryIntent.RELATIONAL
        assert "Mimar Sinan" in result.entities

    def test_temporal_intent(self, analyzer):
        """Test detection of temporal queries."""
        result = analyzer.analyze("Bizans döneminde hangi yapılar yapıldı?")
        assert result.intent == QueryIntent.TEMPORAL
        assert "Bizans" in result.time_references

    def test_factual_intent(self, analyzer):
        """Test detection of factual queries."""
        result = analyzer.analyze("Ayasofya ne zaman yapıldı?")
        assert result.intent == QueryIntent.FACTUAL
        assert "Ayasofya" in result.entities

    def test_comparative_intent(self, analyzer):
        """Test detection of comparative queries."""
        result = analyzer.analyze("Ayasofya ile Sultanahmet Camii arasındaki fark nedir?")
        assert result.intent == QueryIntent.COMPARATIVE
        assert "Ayasofya" in result.entities
        assert "Sultanahmet Camii" in result.entities

    def test_descriptive_intent_default(self, analyzer):
        """Test default descriptive intent."""
        result = analyzer.analyze("Ayasofya hakkında bilgi ver")
        assert result.intent == QueryIntent.DESCRIPTIVE
        assert "Ayasofya" in result.entities

    def test_entity_extraction_ayasofya(self, analyzer):
        """Test entity extraction for Ayasofya."""
        result = analyzer.analyze("Ayasofya'nın tarihi nedir?")
        assert "Ayasofya" in result.entities

    def test_entity_extraction_multiple(self, analyzer):
        """Test extraction of multiple entities."""
        result = analyzer.analyze("Mimar Sinan Sultanahmet Camii'ni mi yaptı?")
        assert "Mimar Sinan" in result.entities
        assert "Sultanahmet Camii" in result.entities

    def test_entity_extraction_dikilitas(self, analyzer):
        """Test entity extraction with Turkish characters."""
        result = analyzer.analyze("Dikilitaş hakkında bilgi ver")
        assert "Dikilitaş" in result.entities

    def test_canonical_name_mapping(self, analyzer):
        """Test that aliases are mapped to canonical names."""
        result = analyzer.analyze("Hagia Sophia hakkında bilgi ver")
        assert "Ayasofya" in result.entities

    def test_time_reference_extraction(self, analyzer):
        """Test time reference extraction."""
        result = analyzer.analyze("537 yılında ne oldu?")
        assert len(result.time_references) > 0

    def test_relationship_type_detection(self, analyzer):
        """Test relationship type detection."""
        result = analyzer.analyze("Kim yaptırdı bu camiyi?")
        assert "YAPTIRAN" in result.relationship_types

    def test_query_analysis_to_dict(self, analyzer):
        """Test QueryAnalysis serialization."""
        result = analyzer.analyze("Ayasofya hakkında bilgi ver")
        result_dict = result.to_dict()

        assert "original_query" in result_dict
        assert "intent" in result_dict
        assert "entities" in result_dict
        assert "confidence" in result_dict


class TestQueryAnalyzerWithMockLLM:
    """Test query analysis with mocked LLM."""

    def test_llm_analysis_success(self, mock_llm):
        """Test LLM-based analysis when it succeeds."""
        # Mock the chain to return valid JSON
        from unittest.mock import Mock, patch

        mock_chain = Mock()
        mock_chain.invoke.return_value = {
            "intent": "origin",
            "entities": ["Dikilitaş"],
            "time_references": [],
            "location_references": ["Mısır"],
            "relationship_types": ["GETIRTEN"],
            "confidence": 0.9
        }

        with patch.object(QueryAnalyzer, '__init__', lambda self, *args, **kwargs: None):
            analyzer = QueryAnalyzer.__new__(QueryAnalyzer)
            analyzer.llm = mock_llm
            analyzer.chain = mock_chain
            analyzer.fallback_to_rules = True

            result = analyzer._llm_analyze("Dikilitaşı kim getirtti?")

            assert result.intent == QueryIntent.ORIGIN
            assert "Dikilitaş" in result.entities
            assert result.confidence == 0.9

    def test_llm_failure_fallback(self, mock_llm):
        """Test fallback to rules when LLM fails."""
        from unittest.mock import Mock

        mock_chain = Mock()
        mock_chain.invoke.side_effect = Exception("API Error")

        analyzer = QueryAnalyzer(llm=None, fallback_to_rules=True)
        analyzer.chain = mock_chain

        result = analyzer.analyze("Ayasofya'nın yanında ne var?")

        # Should fall back to rule-based
        assert result.intent == QueryIntent.SPATIAL
        assert "Ayasofya" in result.entities
