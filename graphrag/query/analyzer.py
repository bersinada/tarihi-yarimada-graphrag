"""Query analysis - intent detection and entity extraction."""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    FACTUAL = "factual"
    RELATIONAL = "relational"
    SPATIAL = "spatial"
    COMPARATIVE = "comparative"
    ORIGIN = "origin"
    TEMPORAL = "temporal"
    DESCRIPTIVE = "descriptive"


@dataclass
class QueryAnalysis:
    original_query: str
    intent: QueryIntent
    entities: List[str] = field(default_factory=list)
    time_references: List[str] = field(default_factory=list)
    location_references: List[str] = field(default_factory=list)
    relationship_types: List[str] = field(default_factory=list)
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "intent": self.intent.value,
            "entities": self.entities,
            "time_references": self.time_references,
            "location_references": self.location_references,
            "relationship_types": self.relationship_types,
            "confidence": self.confidence,
        }


KNOWN_STRUCTURES = [
    "Ayasofya", "Sultanahmet Camii", "Sultanahmet", "Aya Irini",
    u"Dikilita\u015f", "Dikilitas", u"Y\u0131lanl\u0131 S\u00fctun", "Yilanli Sutun",
    u"\u00d6rme Dikilita\u015f", "Orme Dikilitas", u"Alman \u00c7e\u015fmesi", "Alman Cesmesi",
    u"III. Ahmet \u00c7e\u015fmesi", u"III. Ahmed \u00c7e\u015fmesi", u"I. Ahmet T\u00fcrbesi",
    u"Firuz A\u011fa Camii", "Firuzaga Camii", u"Firuza\u011fa Camii",
    u"Topkap\u0131 Saray\u0131", "Topkapi Sarayi", u"Binbirdirek Sarn\u0131c\u0131", "Binbirdirek",
]

KNOWN_PERSONS = [
    "I. Justinianus", "Justinianus",
    "Fatih Sultan Mehmed", "Fatih Sultan Mehmet", "Fatih",
    "I. Ahmet", "Sultan Ahmet", "Mimar Sinan", "Sinan",
    u"Sedefkar Mehmet A\u011fa", "I. Theodosius", "Theodosius",
    "III. Thutmose", "Thutmose", "I. Konstantin", "Konstantin",
    "II. Wilhelm", u"Fossati Karde\u015fler", "Fossati",
]

CANONICAL_MAP = {
    "sultanahmet": "Sultanahmet Camii",
    "mavi cami": "Sultanahmet Camii",
    u"aya irini": u"Aya \u0130rini",
    "dikilitas": u"Dikilita\u015f",
    "yilanli sutun": u"Y\u0131lanl\u0131 S\u00fctun",
    "orme dikilitas": u"\u00d6rme Dikilita\u015f",
    "alman cesmesi": u"Alman \u00c7e\u015fmesi",
    u"iii. ahmed \u00e7e\u015fmesi": u"III. Ahmet \u00c7e\u015fmesi",
    "ahmed cesmesi": u"III. Ahmet \u00c7e\u015fmesi",
    "ahmed turbesi": u"I. Ahmet T\u00fcrbesi",
    "firuzaga camii": u"Firuz A\u011fa Camii",
    u"firuza\u011fa camii": u"Firuz A\u011fa Camii",
    "topkapi sarayi": u"Topkap\u0131 Saray\u0131",
    "binbirdirek": u"Binbirdirek Sarn\u0131c\u0131",
    "justinianus": "I. Justinianus",
    "justinian": "I. Justinianus",
    "fatih": "Fatih Sultan Mehmed",
    "fatih sultan mehmet": "Fatih Sultan Mehmed",
    "sinan": "Mimar Sinan",
    "mimar sinan": "Mimar Sinan",
    "theodosius": "I. Theodosius",
    "thutmose": u"M\u0131s\u0131r Firavunu III. Thutmose",
    "iii. thutmose": u"M\u0131s\u0131r Firavunu III. Thutmose",
    "konstantin": "I. Konstantin",
    "fossati": u"Fossati Karde\u015fler",
    "sultan ahmet": "I. Ahmet",
    "i. ahmed": "I. Ahmet",
}


ANALYSIS_PROMPT = """Sen bir T\u00fcrk\u00e7e soru analizcisisin.
\u0130stanbul Tarihi Yar\u0131mada hakk\u0131ndaki sorular\u0131 analiz ediyorsun.

Soru: {query}

A\u015fa\u011f\u0131daki JSON format\u0131nda cevap ver:
{{
    "intent": "factual|relational|spatial|comparative|origin|temporal|descriptive",
    "entities": ["yap\u0131 veya ki\u015fi isimleri"],
    "time_references": ["tarih veya d\u00f6nem referanslar\u0131"],
    "location_references": ["konum referanslar\u0131"],
    "relationship_types": [],
    "confidence": 0.0-1.0
}}

Intent kurallar\u0131:
- spatial: mesafe, uzakl\u0131k, yak\u0131n\u0131nda, nerede, aras\u0131, konum sorular\u0131
- relational: kim yapt\u0131rd\u0131, kimin \u00f6\u011frencisi, ili\u015fki sorular\u0131
- origin: nereden geldi, nereden ta\u015f\u0131nm\u0131\u015f, k\u00f6ken sorular\u0131
- temporal: d\u00f6nem, y\u00fczy\u0131l, Bizans/Osmanl\u0131 d\u00f6nemi sorular\u0131
- comparative: fark, benzerlik, kar\u015f\u0131la\u015ft\u0131rma sorular\u0131
- factual: ne zaman, ka\u00e7 metre, belirli bilgi sorusu (mesafe HAR\u0130\u00c7)
- descriptive: genel bilgi, hakk\u0131nda bilgi ver

\u00d6NEML\u0130: "mesafe", "uzakl\u0131k", "ne kadar uzak", "aras\u0131" gibi ifadeler MUTLAKA "spatial"!

Bilinen yap\u0131lar: Ayasofya, Sultanahmet Camii, Aya \u0130rini, Dikilita\u015f, Y\u0131lanl\u0131 S\u00fctun,
\u00d6rme Dikilita\u015f, Alman \u00c7e\u015fmesi, III. Ahmet \u00c7e\u015fmesi, I. Ahmet T\u00fcrbesi, Firuz A\u011fa Camii

Sadece JSON format\u0131nda cevap ver."""


class QueryAnalyzer:
    """Kullanici sorgularini analiz eder. LLM + kural tabanli fallback."""

    def __init__(self, llm=None, fallback_to_rules=True):
        self.llm = llm
        self.fallback_to_rules = fallback_to_rules
        if llm:
            self.parser = JsonOutputParser()
            self.prompt = PromptTemplate(
                template=ANALYSIS_PROMPT, input_variables=["query"]
            )
            self.chain = self.prompt | self.llm | self.parser
        else:
            self.chain = None

    def analyze(self, query):
        """Sorguyu analiz et: intent, entity, metadata."""
        if self.chain:
            try:
                result = self._llm_analyze(query)
                if result.confidence >= 0.5:
                    return self._apply_spatial_override(query, result)
            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}")

        if self.fallback_to_rules:
            return self._rule_based_analyze(query)

        return QueryAnalysis(
            original_query=query,
            intent=QueryIntent.DESCRIPTIVE,
            entities=self._extract_entities(query),
            confidence=0.3,
        )

    def _apply_spatial_override(self, query, analysis):
        q = query.lower()
        spatial_kw = [
            "mesafe", "uzakl\u0131k", "uzaklik", "ne kadar uzak",
            "aras\u0131", "arasi", "aras\u0131ndaki mesafe",
        ]
        if any(kw in q for kw in spatial_kw) and analysis.intent != QueryIntent.SPATIAL:
            logger.info(f"Intent override: {analysis.intent.value} -> spatial")
            analysis.intent = QueryIntent.SPATIAL
        return analysis

    def _llm_analyze(self, query):
        result = self.chain.invoke({"query": query})
        try:
            intent = QueryIntent(result.get("intent", "descriptive").lower())
        except ValueError:
            intent = QueryIntent.DESCRIPTIVE
        return QueryAnalysis(
            original_query=query,
            intent=intent,
            entities=result.get("entities", []),
            time_references=result.get("time_references", []),
            location_references=result.get("location_references", []),
            relationship_types=result.get("relationship_types", []),
            confidence=result.get("confidence", 0.7),
        )

    def _rule_based_analyze(self, query):
        q = query.lower()
        return QueryAnalysis(
            original_query=query,
            intent=self._detect_intent(q),
            entities=self._extract_entities(query),
            time_references=self._extract_time_refs(query),
            confidence=0.6,
        )

    def _detect_intent(self, q):
        if any(w in q for w in [
            "mesafe", "uzakl\u0131k", "uzaklik", "ne kadar uzak", "aras\u0131", "arasi",
        ]):
            return QueryIntent.SPATIAL
        if any(w in q for w in [
            "yan\u0131nda", "yaninda", "yak\u0131n\u0131nda", "yakininda",
            "kar\u015f\u0131s\u0131nda", "karsisinda", "nerede", "konumu",
        ]):
            return QueryIntent.SPATIAL
        if any(w in q for w in ["nereden", "nerden", "getir", "geldi", "k\u00f6keni"]):
            return QueryIntent.ORIGIN
        if any(w in q for w in [
            "kim yapt\u0131r", "kim yaptir", "kim in\u015fa", "kim insa",
            "yapt\u0131ran", "yaptiran", "t\u00fcrbesi", "g\u00f6m\u00fcl\u00fc",
        ]):
            return QueryIntent.RELATIONAL
        if any(w in q for w in [
            "d\u00f6neminde", "doneminde", "y\u00fczy\u0131l", "yuzyil",
            "bizans", "osmanl\u0131", "osmanli",
        ]):
            return QueryIntent.TEMPORAL
        if any(w in q for w in ["fark", "kar\u015f\u0131la\u015ft\u0131r", "karsilastir", "hangisi"]):
            return QueryIntent.COMPARATIVE
        if any(w in q for w in ["ne zaman", "ka\u00e7 metre", "kac metre", "tarihi"]):
            return QueryIntent.FACTUAL
        return QueryIntent.DESCRIPTIVE

    def _extract_entities(self, query):
        entities = []
        q_lower = query.lower()
        for name in KNOWN_STRUCTURES + KNOWN_PERSONS:
            if name.lower() in q_lower:
                canonical = CANONICAL_MAP.get(name.lower(), name)
                if canonical not in entities:
                    entities.append(canonical)
        return entities

    def _extract_time_refs(self, query):
        refs = re.findall(r"\b\d{3,4}\b", query)
        for period in ["Bizans", "Osmanl\u0131", "Roma", "Antik"]:
            if period.lower() in query.lower():
                refs.append(period)
        return list(set(refs))
