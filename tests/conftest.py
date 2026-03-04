"""
Pytest fixtures for GraphRAG testing.

Provides mock objects for unit testing without requiring
live Neo4j or API connections.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any


# ==============================================================================
# Mock Neo4j Fixtures
# ==============================================================================

@pytest.fixture
def mock_neo4j_driver():
    """Create a mock Neo4j driver for unit tests."""
    driver = Mock()
    session = MagicMock()

    # Setup context manager
    driver.session.return_value.__enter__ = Mock(return_value=session)
    driver.session.return_value.__exit__ = Mock(return_value=False)

    return driver, session


@pytest.fixture
def mock_neo4j_client(mock_neo4j_driver):
    """Create a mock Neo4jClient."""
    from graphrag.database.neo4j_client import Neo4jClient

    driver, session = mock_neo4j_driver

    with patch.object(Neo4jClient, '__init__', lambda self, *args, **kwargs: None):
        client = Neo4jClient.__new__(Neo4jClient)
        client._driver = driver
        client.uri = "bolt://localhost:7687"
        client.username = "neo4j"

        # Mock execute_query
        client.execute_query = Mock(return_value=[])
        client.execute_write = Mock(return_value={"nodes_created": 0})

        return client


# ==============================================================================
# Mock Embedder Fixtures
# ==============================================================================

@pytest.fixture
def mock_embedder():
    """Create a mock embedder for unit tests."""
    embedder = Mock()
    embedder.dimension = 384
    embedder.model_name = "mock-model"

    # Return consistent embeddings
    embedder.embed_text.return_value = [0.1] * 384
    embedder.embed_batch.return_value = [[0.1] * 384]

    return embedder


@pytest.fixture
def real_embedder():
    """Create a real SentenceTransformer embedder (for integration tests)."""
    try:
        from graphrag.embeddings.sentence_transformer import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder()
    except ImportError:
        pytest.skip("sentence-transformers not installed")


# ==============================================================================
# Sample Data Fixtures
# ==============================================================================

@pytest.fixture
def sample_query_analysis():
    """Sample query analysis result."""
    from graphrag.query.analyzer import QueryAnalysis, QueryIntent

    return QueryAnalysis(
        original_query="Dikilitaşı Mısır'dan kim getirtti?",
        intent=QueryIntent.ORIGIN,
        entities=["Dikilitaş"],
        time_references=[],
        location_references=["Mısır"],
        relationship_types=["GETIRTEN", "TASINDI_BURADAN"],
        confidence=0.9
    )


@pytest.fixture
def sample_graph_data():
    """Sample graph data representing the knowledge graph."""
    return {
        "nodes": [
            {"id": "Dikilitaş", "label": "Yapi", "insa_tarihi": "MÖ 15. yüzyıl", "yukseklik_metre": 18.45},
            {"id": "III. Thutmose", "label": "Firavun"},
            {"id": "I. Theodosius", "label": "Imparator"},
            {"id": "Karnak Tapınağı", "label": "Lokasyon"},
            {"id": "Ayasofya", "label": "Yapi", "insa_tarihi": "537"},
            {"id": "I. Justinianus", "label": "Imparator"},
            {"id": "Sultanahmet Camii", "label": "Yapi", "insa_tarihi": "1616"},
            {"id": "I. Ahmed", "label": "Padisah"},
            {"id": "Sedefkâr Mehmed Ağa", "label": "Mimar"},
            {"id": "Mimar Sinan", "label": "Mimar"},
        ],
        "relationships": [
            ("Dikilitaş", "YAPTIRAN", "III. Thutmose"),
            ("Dikilitaş", "GETIRTEN", "I. Theodosius"),
            ("Dikilitaş", "TASINDI_BURADAN", "Karnak Tapınağı"),
            ("Ayasofya", "YAPTIRAN", "I. Justinianus"),
            ("Sultanahmet Camii", "YAPTIRAN", "I. Ahmed"),
            ("Sultanahmet Camii", "TASARLAYAN", "Sedefkâr Mehmed Ağa"),
            ("Sedefkâr Mehmed Ağa", "OGRENCISI", "Mimar Sinan"),
            ("Ayasofya", "KARSISINDA", "Sultanahmet Camii"),
        ]
    }


@pytest.fixture
def sample_vector_results():
    """Sample vector search results."""
    from graphrag.retrieval.vector_retriever import VectorSearchResult

    return [
        VectorSearchResult(
            node_id="1",
            label="Yapi",
            properties={"id": "Ayasofya", "insa_tarihi": "537"},
            score=0.95,
            text="Ayasofya"
        ),
        VectorSearchResult(
            node_id="2",
            label="Yapi",
            properties={"id": "Sultanahmet Camii", "insa_tarihi": "1616"},
            score=0.85,
            text="Sultanahmet Camii"
        ),
        VectorSearchResult(
            node_id="3",
            label="Document",
            properties={"content": "Ayasofya 537 yılında inşa edilmiştir.", "source_file": "ayasofya.txt"},
            score=0.80,
            text="Ayasofya tarihi bilgisi"
        ),
    ]


@pytest.fixture
def sample_graph_results():
    """Sample graph search results."""
    from graphrag.retrieval.graph_retriever import GraphSearchResult

    return [
        GraphSearchResult(
            source_entity="Ayasofya",
            target_entity="I. Justinianus",
            relationship="YAPTIRAN",
            direction="outgoing",
            path_length=1,
            properties={},
            context="Ayasofya I. Justinianus tarafından yaptırıldı"
        ),
        GraphSearchResult(
            source_entity="Sultanahmet Camii",
            target_entity="Ayasofya",
            relationship="KARSISINDA",
            direction="bidirectional",
            path_length=1,
            properties={},
            context="Sultanahmet Camii Ayasofya'nın karşısındadır"
        ),
    ]


# ==============================================================================
# Mock LLM Fixtures
# ==============================================================================

@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock()

    # Mock response
    response = Mock()
    response.content = "Ayasofya, I. Justinianus tarafından 537 yılında yaptırılmıştır."

    llm.invoke.return_value = response

    return llm


@pytest.fixture
def mock_groq_llm(mock_llm):
    """Create a mock Groq LLM."""
    with patch('langchain_groq.ChatGroq', return_value=mock_llm):
        yield mock_llm


# ==============================================================================
# Configuration Fixtures
# ==============================================================================

@pytest.fixture
def test_config():
    """Create test configuration."""
    from graphrag.config import Config, Neo4jConfig, EmbeddingsConfig, LLMConfig

    return Config(
        neo4j=Neo4jConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="test_password"
        ),
        embeddings=EmbeddingsConfig(
            provider="sentence_transformer",
            model="paraphrase-multilingual-MiniLM-L12-v2",
            dimension=384
        ),
        llm=LLMConfig(
            provider="groq",
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=2048
        )
    )


# ==============================================================================
# Integration Test Markers
# ==============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires live services)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# ==============================================================================
# Skip Conditions
# ==============================================================================

@pytest.fixture
def skip_if_no_neo4j():
    """Skip test if Neo4j is not available."""
    import os
    if os.getenv("SKIP_NEO4J_TESTS", "1") == "1":
        pytest.skip("Neo4j tests disabled")


@pytest.fixture
def skip_if_no_groq():
    """Skip test if Groq API key is not available."""
    import os
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")
