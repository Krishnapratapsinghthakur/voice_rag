#!/usr/bin/env python3
"""
============================================
PDF Ingestion Script for Voice AI Assistant
============================================

This script loads a PDF document, splits it into chunks,
generates embeddings using Google's embedding model, and
persists them to a local ChromaDB vector store.

Usage:
    1. Place your PDF file at data/source.pdf
    2. Set GOOGLE_API_KEY environment variable
    3. Run: python ingest.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ============================================
# Configuration
# ============================================

# Load environment variables from .env file
load_dotenv()

# Paths
PDF_PATH = Path("data/source.pdf")
CHROMA_DB_PATH = Path("chroma_db")

# Chunking parameters (optimized for RAG)
CHUNK_SIZE = 1000       # Characters per chunk
CHUNK_OVERLAP = 200     # 20% overlap for context continuity

# Collection name in ChromaDB
COLLECTION_NAME = "pdf_documents"


def validate_environment() -> None:
    """
    Validates that all required environment variables and files are present.
    Exits with an error message if validation fails.
    """
    # Check for Google API Key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY environment variable is not set.")
        print("   Please set it using: export GOOGLE_API_KEY='your_key_here'")
        print("   Or create a .env file with: GOOGLE_API_KEY=your_key_here")
        sys.exit(1)
    
    # Check for PDF file
    if not PDF_PATH.exists():
        print(f"❌ Error: PDF file not found at '{PDF_PATH}'")
        print(f"   Please place your PDF document at: {PDF_PATH.absolute()}")
        sys.exit(1)
    
    print("✅ Environment validation passed")


def load_pdf(pdf_path: Path) -> list:
    """
    Loads a PDF file and returns a list of Document objects.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of LangChain Document objects (one per page)
    """
    print(f"📄 Loading PDF: {pdf_path}")
    
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    
    print(f"   Loaded {len(documents)} pages")
    return documents


def split_documents(documents: list) -> list:
    """
    Splits documents into smaller chunks for better embedding quality.
    
    Uses RecursiveCharacterTextSplitter which tries to split on
    natural boundaries (paragraphs, sentences, words) before
    falling back to character-level splits.
    
    Args:
        documents: List of Document objects
        
    Returns:
        List of chunked Document objects
    """
    print(f"✂️  Splitting documents (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # Priority order for splitting
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"   Created {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks: list) -> Chroma:
    """
    Creates embeddings for all chunks and stores them in ChromaDB.
    
    Uses HuggingFace's all-MiniLM-L6-v2 model for generating embeddings.
    This runs locally and doesn't require any API calls.
    
    Args:
        chunks: List of Document chunks
        
    Returns:
        Chroma vector store instance
    """
    print("🧠 Creating embeddings and vector store...")
    print("   Using model: all-MiniLM-L6-v2 (local)")
    
    # Initialize local embedding model (no API needed)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create and persist vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DB_PATH)
    )
    
    print(f"   Vector store saved to: {CHROMA_DB_PATH.absolute()}")
    return vectorstore


def main():
    """
    Main ingestion pipeline.
    """
    print("\n" + "=" * 50)
    print("🚀 PDF Ingestion Pipeline")
    print("=" * 50 + "\n")
    
    # Step 1: Validate environment
    validate_environment()
    
    # Step 2: Load PDF
    documents = load_pdf(PDF_PATH)
    
    # Step 3: Split into chunks
    chunks = split_documents(documents)
    
    # Step 4: Create vector store
    vectorstore = create_vector_store(chunks)
    
    # Summary
    print("\n" + "=" * 50)
    print("✅ Ingestion Complete!")
    print("=" * 50)
    print(f"   📊 Total chunks ingested: {len(chunks)}")
    print(f"   📁 Vector store location: {CHROMA_DB_PATH.absolute()}")
    print("\n   You can now run: python app.py")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
