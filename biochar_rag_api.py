#!/usr/bin/env python3
"""
Biochar RAG API with LLAMA Llama-3.2-3B-Instruct-Turbo"  # Available model on Together AI
===================================================

FastAPI service that provides RAG functionality using:
- LLAMA Llama-3.2-3B-Instruct-Turbo model via Together AI
- Biochar corpus embeddings for retrieval
- CPU-optimized deployment for Render

Environment Variables Required:
- TOGETHER_API_KEY: Together AI API key
- EMBEDDING_MODEL: sentence-transformers model name (optional)
- MAX_CONTEXT_CHUNKS: Maximum chunks to retrieve (optional)
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime

# Import our download utility
from download_embeddings import download_and_extract_embeddings

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed - use system environment variables
    pass

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# ML libraries - temporarily commented out for basic deployment
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# Temporary imports for basic deployment
import warnings
warnings.filterwarnings("ignore")

# Together AI integration
try:
    from together import Together
    together_available = True
except ImportError:
    Together = None
    together_available = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    # API Configuration
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")  # Match notebook model
    MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))
    
    # Model Configuration
    DEEPSEEK_MODEL = "meta-llama/Llama-3.2-3B-Instruct-Turbo"  # Available model on Together AI
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    
    # File paths
    EMBEDDINGS_DIR = Path("embeddings_output")
    EMBEDDINGS_FILE = EMBEDDINGS_DIR / "biochar_embeddings.npy"
    CHUNKS_FILE = EMBEDDINGS_DIR / "biochar_chunks_with_embeddings.csv"
    METADATA_FILE = EMBEDDINGS_DIR / "embedding_metadata.json"

# Biochar terminology and acronym expansion
BIOCHAR_TERMINOLOGY = {
    "TLUD": "Top-Lit Updraft (TLUD)",
    "tlud": "top-lit updraft (TLUD)",
    "TLUD stove": "Top-Lit Updraft (TLUD) stove",
    "tlud stove": "top-lit updraft (TLUD) stove",
    "TLUD gasifier": "Top-Lit Updraft (TLUD) gasifier",
    "tlud gasifier": "top-lit updraft (TLUD) gasifier",
    "HTC": "Hydrothermal Carbonization (HTC)",
    "htc": "hydrothermal carbonization (HTC)",
    "CEC": "Cation Exchange Capacity (CEC)",
    "cec": "cation exchange capacity (CEC)",
    "SOM": "Soil Organic Matter (SOM)",
    "som": "soil organic matter (SOM)",
    "WHC": "Water Holding Capacity (WHC)",
    "whc": "water holding capacity (WHC)",
    "BET": "Brunauer-Emmett-Teller (BET) surface area",
    "bet": "Brunauer-Emmett-Teller (BET) surface area",
    "IBI": "International Biochar Initiative (IBI)",
    "ici": "international biochar initiative (IBI)"
}

def expand_biochar_terminology(text: str) -> str:
    """Expand biochar acronyms and terminology for better understanding."""
    expanded_text = text
    
    # Replace acronyms with expanded forms
    for acronym, expansion in BIOCHAR_TERMINOLOGY.items():
        # Use word boundaries to avoid partial replacements
        import re
        pattern = r'\b' + re.escape(acronym) + r'\b'
        expanded_text = re.sub(pattern, expansion, expanded_text)
    
    return expanded_text

# Pydantic models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's question about biochar")
    max_chunks: Optional[int] = Field(Config.MAX_CONTEXT_CHUNKS, description="Max context chunks to retrieve")
    include_sources: Optional[bool] = Field(True, description="Include source information in response")

class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    similarity: float
    tokens: int

class ChatResponse(BaseModel):
    response: str
    retrieved_chunks: List[RetrievedChunk]
    model_used: str
    processing_time_ms: int

class HealthResponse(BaseModel):
    status: str
    version: str
    embeddings_loaded: bool
    model_loaded: bool
    total_chunks: int

# Initialize FastAPI app
app = FastAPI(
    title="Biochar RAG API",
    description="RAG service for biochar research using DeepSeek V2 Chat",
    version="1.0.0"
)

# Enable CORS for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and data
embedding_model = None
chunks_df = None
embeddings = None
metadata = None

class RAGService:
    """Core RAG service combining retrieval and generation."""
    
    def __init__(self):
        self.embedding_model = None
        self.chunks_df = None
        self.embeddings = None
        self.metadata = None
        self.together_client = None
        
    def load_data(self):
        """Load embeddings and chunks data. Download from cloud if needed."""
        try:
            # Temporarily return success without loading data
            logger.info("Mock data loading - skipping actual file operations")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
            
            # Load metadata
            if Config.METADATA_FILE.exists():
                with open(Config.METADATA_FILE, 'r') as f:
                    self.metadata = json.load(f)
                logger.info("Loaded embedding metadata")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
    
    def load_models(self):
        """Load embedding model and initialize Together AI client."""
        try:
            # Temporarily disable ML models for basic deployment
            # self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL, device='cpu')
            # logger.info(f"Loaded embedding model: {Config.EMBEDDING_MODEL}")
            
            # Initialize Together AI client
            if not Config.TOGETHER_API_KEY:
                logger.warning("TOGETHER_API_KEY not set - chat functionality disabled")
                return False
                
            if not together_available:
                logger.error("Together AI library not installed")
                return False
                
            self.together_client = Together(api_key=Config.TOGETHER_API_KEY)
            logger.info("Initialized Together AI client")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def retrieve_relevant_chunks(self, query: str, max_chunks: int = Config.MAX_CONTEXT_CHUNKS) -> List[Dict]:
        """Retrieve most relevant chunks for a query."""
        # Temporarily return mock data for basic deployment
        logger.info(f"Mock retrieval for query: {query}")
        
        # Return mock chunks
        mock_chunks = [
            {
                'chunk_id': f'mock_chunk_{i}',
                'doc_id': f'mock_doc_{i}',
                'content': f'Mock content for query: {query} (chunk {i})',
                'similarity': 0.8 - (i * 0.1),
                'tokens': 100
            }
            for i in range(min(max_chunks, 3))
        ]
        
        return mock_chunks
    
    def generate_response(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate response using DeepSeek V2 Chat with retrieved context."""
        if self.together_client is None:
            raise HTTPException(status_code=500, detail="Together AI client not initialized")
        
        # Build context from retrieved chunks
        context_text = "\n\n".join([
            f"[Source: {chunk['doc_id']}]\n{chunk['content']}"
            for chunk in context_chunks
        ])
        
        # Create system prompt
        system_prompt = """You are a helpful assistant specializing in biochar research and applications. 
You have access to a comprehensive database of biochar-related documents, research papers, and technical information.

Guidelines:
- Answer questions accurately based on the provided context
- If the context doesn't contain relevant information, say so clearly
- Cite specific sources when possible using the [Source: filename] format
- Provide practical, science-based information
- Be concise but thorough in your explanations"""

        # Create user prompt with context
        user_prompt = f"""Context from biochar research database:

{context_text}

Question: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain sufficient information to answer the question, please indicate that and suggest what additional information might be needed."""

        # Prepare messages for Together AI
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Call DeepSeek V2 Chat via Together AI
            response = self.together_client.chat.completions.create(
                model=Config.DEEPSEEK_MODEL,
                messages=messages,
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Together AI API error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

# Initialize service
rag_service = RAGService()

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG service on startup."""
    logger.info("Starting Biochar RAG API...")
    
    # Load data and models
    data_loaded = rag_service.load_data()
    models_loaded = rag_service.load_models()
    
    if not data_loaded:
        logger.error("Failed to load embeddings and chunks data")
    
    if not models_loaded:
        logger.error("Failed to load models - some functionality may be disabled")
    
    logger.info("Biochar RAG API startup complete")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        embeddings_loaded=rag_service.embeddings is not None,
        model_loaded=rag_service.embedding_model is not None,
        total_chunks=len(rag_service.chunks_df) if rag_service.chunks_df is not None else 0
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with RAG functionality."""
    start_time = datetime.now()
    
    try:
        # Expand biochar terminology and acronyms
        expanded_message = expand_biochar_terminology(request.message)
        logger.info(f"Original: {request.message}")
        logger.info(f"Expanded: {expanded_message}")
        
        # Retrieve relevant chunks using expanded message
        retrieved_chunks = rag_service.retrieve_relevant_chunks(
            expanded_message, 
            request.max_chunks
        )
        
        # Generate response using expanded message
        response_text = rag_service.generate_response(expanded_message, retrieved_chunks)
        
        # Format retrieved chunks for response
        chunks_response = [
            RetrievedChunk(
                chunk_id=chunk['chunk_id'],
                doc_id=chunk['doc_id'],
                content=chunk['content'][:500] + "..." if len(chunk['content']) > 500 else chunk['content'],
                similarity=chunk['similarity'],
                tokens=chunk['tokens']
            )
            for chunk in retrieved_chunks
        ]
        
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return ChatResponse(
            response=response_text,
            retrieved_chunks=chunks_response if request.include_sources else [],
            model_used=Config.DEEPSEEK_MODEL,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_chunks(
    query: str = Query(..., description="Search query"),
    max_results: int = Query(10, description="Maximum results to return")
):
    """Search for relevant chunks without generating a chat response."""
    try:
        retrieved_chunks = rag_service.retrieve_relevant_chunks(query, max_results)
        return {
            "query": query,
            "results": retrieved_chunks,
            "total_found": len(retrieved_chunks)
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get statistics about the knowledge base."""
    if rag_service.chunks_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    stats = {
        "total_chunks": len(rag_service.chunks_df),
        "total_documents": rag_service.chunks_df['doc_id'].nunique(),
        "average_chunk_tokens": float(rag_service.chunks_df['chunk_tokens'].mean()),
        "document_types": rag_service.chunks_df['doc_type'].value_counts().to_dict() if 'doc_type' in rag_service.chunks_df.columns else {},
        "embedding_dimension": rag_service.embeddings.shape[1] if rag_service.embeddings is not None else 0,
        "model_info": {
            "embedding_model": Config.EMBEDDING_MODEL,
            "chat_model": Config.DEEPSEEK_MODEL
        }
    }
    
    if rag_service.metadata:
        stats["metadata"] = rag_service.metadata
    
    return stats

if __name__ == "__main__":
    # For local development
    uvicorn.run(
        "biochar_rag_api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True
    )

# Swagger UI documentation available at:
# http://localhost:8000/docs