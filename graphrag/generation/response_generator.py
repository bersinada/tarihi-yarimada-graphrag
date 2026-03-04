"""Response generation using LLM with hybrid retrieval context."""

import logging
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from ..retrieval.hybrid_retriever import HybridSearchResult
from . import prompt_templates

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """LLM ile cevap uretici. Graf + vektor baglamini birlestirip Turkce cevap uretir."""

    def __init__(
        self,
        llm: ChatGoogleGenerativeAI,
        max_context_chars: int = 4000,
        max_graph_contexts: int = 15,
        max_vector_results: int = 5,
    ):
        self.llm = llm
        self.max_context_chars = max_context_chars
        self.max_graph_contexts = max_graph_contexts
        self.max_vector_results = max_vector_results

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(prompt_templates.SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(prompt_templates.RAG_PROMPT_TEMPLATE),
        ])
        self.chain = self.prompt | self.llm

    def generate(
        self,
        query: str,
        results: List[HybridSearchResult],
        include_sources: bool = False,
        spatial_summary: str = "",
    ) -> str:
        """Hybrid retrieval sonuclariyla cevap uret."""
        if not results:
            return prompt_templates.NO_CONTEXT_TEMPLATE.format(query=query)

        graph_context = self._build_graph_context(results)
        vector_context = self._build_vector_context(results)

        if spatial_summary:
            graph_context += f"\n\n--- Uzamsal Analiz ---\n{spatial_summary}"

        if not graph_context.strip() and not vector_context.strip():
            return prompt_templates.NO_CONTEXT_TEMPLATE.format(query=query)

        try:
            response = self.chain.invoke({
                "graph_context": graph_context or "(Graf iliskisi bulunamadi)",
                "vector_context": vector_context or "(Semantik benzerlik bulunamadi)",
                "query": query,
            })
            return response.content
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"Yanit olusturulurken bir hata olustu: {str(e)}"

    def _build_graph_context(self, results: List[HybridSearchResult]) -> str:
        """Graf baglamini formatla."""
        context_parts = []
        seen_entities = set()
        total_chars = 0

        for result in results:
            if result.entity in seen_entities:
                continue
            seen_entities.add(result.entity)

            graph_contexts = result.metadata.get("graph_contexts", [])
            if not graph_contexts:
                continue

            unique_contexts = list(dict.fromkeys(graph_contexts))[:self.max_graph_contexts]
            relationships = "\n".join([f"  - {ctx}" for ctx in unique_contexts])

            props = result.metadata.get("properties", {})
            properties_list = []

            if "latitude" in props and "longitude" in props:
                lat, lon = props.get("latitude"), props.get("longitude")
                if lat and lon:
                    properties_list.append(f"  - Koordinatlar: {lat}K, {lon}D")

            for k, v in props.items():
                if k not in ("embedding", "id", "latitude", "longitude") and v:
                    properties_list.append(f"  - {k}: {v}")

            properties = "\n".join(properties_list) if properties_list else "  (ozellik yok)"

            context_block = prompt_templates.GRAPH_CONTEXT_TEMPLATE.format(
                entity=result.entity,
                label=result.label,
                relationships=relationships,
                properties=properties,
            )

            if total_chars + len(context_block) > self.max_context_chars:
                break

            context_parts.append(context_block)
            total_chars += len(context_block)

        return "\n".join(context_parts)

    def _build_vector_context(self, results: List[HybridSearchResult]) -> str:
        """Vektor baglamini formatla."""
        context_parts = []
        vector_count = 0
        seen_content = set()

        for result in results:
            if not result.vector_score or vector_count >= self.max_vector_results:
                continue

            content = result.content
            if not content:
                continue

            content_key = content[:100]
            if content_key in seen_content:
                continue
            seen_content.add(content_key)

            entity_name = result.entity
            if "_chunk_" in entity_name:
                entity_name = entity_name.split("_chunk_")[0]

            if len(content) > 400:
                content = content[:400] + "..."

            context_parts.append(prompt_templates.VECTOR_CONTEXT_TEMPLATE.format(
                score=result.vector_score,
                entity=entity_name,
                content=content,
            ))
            vector_count += 1

        return "\n".join(context_parts)
