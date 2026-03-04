"""
GraphRAG REST API - FastAPI server for the Istanbul Historical Peninsula chatbot.

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8002

API Endpoints:
    POST /query     - Submit a question and get an answer
    GET  /status    - Get system status
    GET  /health    - Health check
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from graphrag.facade import GraphRAGFacade

logger = logging.getLogger(__name__)

# Global GraphRAG instance
rag: Optional[GraphRAGFacade] = None


# --- Pydantic Models ---

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., min_length=1, max_length=500, description="Soru metni")
    alpha: Optional[float] = Field(None, ge=0.0, le=1.0, description="Vektör/graf dengesi (0-1)")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Ayasofya'yı kim yaptırdı?",
                "alpha": 0.5
            }
        }


class QueryAnalysisResponse(BaseModel):
    """Analysis details in the response."""
    intent: str
    entities: List[str]
    confidence: float


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    success: bool
    query: str
    response: str
    analysis: QueryAnalysisResponse
    sources: List[str]
    metadata: Dict[str, Any]

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "query": "Ayasofya'yı kim yaptırdı?",
                "response": "Ayasofya, Bizans İmparatoru I. Justinianus tarafından 532-537 yılları arasında yaptırılmıştır.",
                "analysis": {
                    "intent": "builder",
                    "entities": ["Ayasofya"],
                    "confidence": 0.95
                },
                "sources": ["Ayasofya"],
                "metadata": {"result_count": 5}
            }
        }


class StatusResponse(BaseModel):
    """Response model for status endpoint."""
    neo4j_connected: bool
    node_counts: Dict[str, int]
    vector_indexes: List[Dict[str, Any]]
    document_stats: Dict[str, Any]


class HealthResponse(BaseModel):
    """Response model for health endpoint."""
    status: str
    service: str


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage GraphRAG lifecycle."""
    global rag
    logger.info("Starting GraphRAG API...")

    try:
        rag = GraphRAGFacade()
        logger.info("GraphRAG initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize GraphRAG: {e}")
        raise

    yield

    # Cleanup
    if rag:
        rag.close()
        logger.info("GraphRAG closed")


# --- FastAPI App ---

app = FastAPI(
    title="İstanbul Tarihi Yarımada GraphRAG API",
    description="""
    Tarihi Yarımada hakkında sorularınızı yanıtlayan hibrit graf+vektör tabanlı soru-cevap sistemi.

    ## Örnek Sorular
    - Ayasofya'yı kim yaptırdı?
    - Dikilitaş nereden getirildi?
    - Sultanahmet Camii'nin yanında ne var?
    - Mimar Sinan'ın eserleri nelerdir?
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
# Production'da allow_origins listesini kısıtlayın, örn:
# allow_origins=["https://sizin-siteniz.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="graphrag-api"
    )


@app.get("/status", response_model=StatusResponse, tags=["System"])
async def get_status():
    """Get system status and statistics."""
    if not rag:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")

    try:
        status = rag.get_system_status()
        return StatusResponse(
            neo4j_connected=status.get("neo4j_connected", False),
            node_counts=status.get("node_counts", {}),
            vector_indexes=status.get("vector_indexes", []),
            document_stats=status.get("document_stats", {})
        )
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """
    Submit a question about Istanbul Historical Peninsula.

    Returns an AI-generated answer based on the knowledge graph and documents.
    """
    if not rag:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")

    try:
        # Optionally set alpha
        if request.alpha is not None:
            rag.set_retrieval_alpha(request.alpha)

        # Process query
        result = rag.query(request.query)

        return QueryResponse(
            success=True,
            query=result.query,
            response=result.response,
            analysis=QueryAnalysisResponse(
                intent=result.analysis.intent.value,
                entities=result.analysis.entities,
                confidence=result.analysis.confidence
            ),
            sources=result.sources,
            metadata=result.metadata
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API info."""
    return {
        "service": "İstanbul Tarihi Yarımada GraphRAG API",
        "version": "1.1.0",
        "docs": "/docs",
        "endpoints": {
            "query": "POST /query",
            "status": "GET /status",
            "health": "GET /health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
