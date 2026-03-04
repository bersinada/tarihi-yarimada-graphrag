"""
Configuration management for GraphRAG system.
Loads settings from YAML file with environment variable interpolation.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Neo4jConfig(BaseModel):
    """Neo4j database configuration."""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str


class EmbeddingsConfig(BaseModel):
    """Embedding model configuration."""
    provider: str = "sentence_transformer"
    model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    dimension: int = 384


class VectorIndexConfig(BaseModel):
    """Vector index configuration."""
    name_prefix: str = "embedding"
    similarity_function: str = "cosine"
    indexed_labels: List[str] = Field(default_factory=lambda: [
        "Structure", "Building", "Person", "Location", "Document"
    ])


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: str = "google"
    model: str = "gemini-2.5-flash"
    temperature: float = 0.1
    max_tokens: int = 2048


class RetrievalConfig(BaseModel):
    """Retrieval strategy configuration."""
    vector_top_k: int = 10
    graph_max_hops: int = 3
    hybrid_alpha: float = 0.5
    min_similarity: float = 0.4
    rrf_k: int = 60


class QueryConfig(BaseModel):
    """Query analysis configuration."""
    intent_classification: bool = True
    entity_extraction: bool = True
    fallback_to_rules: bool = True


class DocumentsConfig(BaseModel):
    """Document processing configuration."""
    source_dir: str = "son-veri"
    chunk_size: int = 500
    chunk_overlap: int = 50


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class Config(BaseModel):
    """Main configuration container."""
    neo4j: Neo4jConfig
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    vector_index: VectorIndexConfig = Field(default_factory=VectorIndexConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    documents: DocumentsConfig = Field(default_factory=DocumentsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, config_path: str = "config.yaml") -> "Config":
        """
        Load configuration from YAML file with environment variable interpolation.

        Supports ${VAR} and ${VAR:default} syntax for environment variables.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Config instance
        """
        # Load .env file if it exists
        from dotenv import load_dotenv
        load_dotenv()

        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, 'r', encoding='utf-8') as f:
            raw_config = f.read()

        # Interpolate environment variables
        interpolated = cls._interpolate_env_vars(raw_config)

        # Parse YAML
        config_dict = yaml.safe_load(interpolated)

        return cls(**config_dict)

    @staticmethod
    def _interpolate_env_vars(text: str) -> str:
        """
        Replace ${VAR} and ${VAR:default} patterns with environment values.

        Args:
            text: Text containing variable references

        Returns:
            Text with variables replaced
        """
        # Pattern: ${VAR} or ${VAR:default_value}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2)

            value = os.environ.get(var_name)

            if value is not None:
                return value
            elif default_value is not None:
                return default_value
            else:
                raise ValueError(f"Environment variable {var_name} not set and no default provided")

        return re.sub(pattern, replacer, text)


def get_config(config_path: str = "config.yaml") -> Config:
    """
    Get configuration singleton.

    Args:
        config_path: Path to configuration file

    Returns:
        Config instance
    """
    return Config.load(config_path)
