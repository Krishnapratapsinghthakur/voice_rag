#!/usr/bin/env python3
"""
============================================
FastAPI Backend for Voice AI Assistant
============================================

Exposes the RAG pipeline via REST API endpoints.

Endpoints:
    GET  /health  - Health check
    POST /query   - Send question, get RAG response
    POST /tts     - Convert text to speech audio
"""

import os
import io
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from gtts import gTTS

# ============================================
# Configuration
# ============================================

load_dotenv()

# Paths (relative to project root)
CHROMA_DB_PATH = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "pdf_documents"

# Gemini Configuration
GEMINI_MODEL = "gemini-1.5-flash"
TEMPERATURE = 0.3

# RAG Configuration
RETRIEVAL_K = 3

# ============================================
# Prompt Template
# ============================================

SYSTEM_PROMPT = """You are a helpful voice assistant that answers questions based on the provided context.

IMPORTANT RULES:
1. Only answer based on the provided context
2. If the context doesn't contain the answer, say "I don't have information about that in my knowledge base"
3. Keep responses concise and natural for voice output (2-3 sentences max)
4. Don't use markdown, bullet points, or special formatting
5. Speak naturally as if having a conversation

Context from knowledge base:
{context}

User question: {question}

Provide a helpful, conversational response:"""

# ============================================
# Request/Response Models
# ============================================

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]

class TTSRequest(BaseModel):
    text: str

# ============================================
# Global State
# ============================================

vectorstore = None
llm = None

# ============================================
# Lifespan Handler
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, cleanup on shutdown."""
    global vectorstore, llm
    
    print("🔧 Initializing backend...")
    
    # Validate API key
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY environment variable not set")
    
    # Validate ChromaDB exists
    if not CHROMA_DB_PATH.exists():
        raise RuntimeError(f"ChromaDB not found at {CHROMA_DB_PATH}. Run 'python ingest.py' first")
    
    # Initialize local embeddings (no API needed)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Load vector store
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DB_PATH),
        embedding_function=embeddings
    )
    print(f"📚 Vector store loaded ({vectorstore._collection.count()} documents)")
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=TEMPERATURE,
        convert_system_message_to_human=True
    )
    print(f"🤖 Gemini LLM initialized ({GEMINI_MODEL})")
    
    print("✅ Backend ready!\n")
    
    yield  # App runs here
    
    # Cleanup
    print("\n🔌 Shutting down...")

# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="Voice AI Assistant API",
    description="RAG-powered question answering with TTS",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative React port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Endpoints
# ============================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "vectorstore_loaded": vectorstore is not None,
        "llm_loaded": llm is not None
    }


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG pipeline.
    
    Retrieves relevant context from ChromaDB and generates
    a response using Gemini.
    """
    if not vectorstore or not llm:
        raise HTTPException(status_code=503, detail="Backend not initialized")
    
    try:
        # Retrieve relevant documents
        docs = vectorstore.similarity_search(request.question, k=RETRIEVAL_K)
        
        if not docs:
            return QueryResponse(
                question=request.question,
                answer="I don't have any information about that in my knowledge base.",
                sources=[]
            )
        
        # Build context
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        sources = list(set([
            doc.metadata.get("source", "Unknown") 
            for doc in docs
        ]))
        
        # Generate response
        prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
        chain = prompt | llm | StrOutputParser()
        
        answer = chain.invoke({
            "context": context,
            "question": request.question
        })
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech.
    
    Returns MP3 audio as a streaming response.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Generate speech
        tts = gTTS(text=request.text, lang='en', slow=False)
        
        # Save to bytes buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return StreamingResponse(
            audio_buffer,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=response.mp3"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


# ============================================
# Run with: uvicorn backend.main:app --reload
# ============================================
