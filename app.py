#!/usr/bin/env python3
"""
============================================
Voice AI Assistant with RAG
============================================

A voice-based assistant that:
1. Listens to your microphone
2. Transcribes speech to text
3. Searches a local knowledge base (PDF via ChromaDB)
4. Generates responses using Google Gemini
5. Speaks the answer back to you

Usage:
    1. First run: python ingest.py (to build the knowledge base)
    2. Set GOOGLE_API_KEY environment variable
    3. Run: python app.py
    4. Say "exit" or "stop" to quit
"""

import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# Speech Recognition
import speech_recognition as sr

# Text-to-Speech
from gtts import gTTS

# Audio Playback
import pygame

# LangChain & Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============================================
# Configuration
# ============================================

# Load environment variables
load_dotenv()

# Paths
CHROMA_DB_PATH = Path("chroma_db")
COLLECTION_NAME = "pdf_documents"

# Gemini Model Configuration
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"  # Latest Gemini 2.5 Flash
TEMPERATURE = 0.3                   # Lower = more focused responses

# Speech Recognition Configuration
LISTEN_TIMEOUT = 5      # Seconds to wait for speech to start
PHRASE_TIME_LIMIT = 15  # Maximum duration of a single phrase

# RAG Configuration
RETRIEVAL_K = 3  # Number of relevant chunks to retrieve

# Exit commands (case-insensitive)
EXIT_COMMANDS = {"exit", "stop", "quit", "bye", "goodbye"}

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
# Initialization Functions
# ============================================

def validate_environment() -> bool:
    """
    Validates that all required components are available.
    
    Returns:
        True if validation passes, False otherwise
    """
    errors = []
    
    # Check Google API Key
    if not os.getenv("GOOGLE_API_KEY"):
        errors.append("GOOGLE_API_KEY environment variable is not set")
    
    # Check ChromaDB exists
    if not CHROMA_DB_PATH.exists():
        errors.append(f"Vector store not found at '{CHROMA_DB_PATH}'. Run 'python ingest.py' first")
    
    if errors:
        print("\n❌ Validation Errors:")
        for error in errors:
            print(f"   • {error}")
        return False
    
    print("✅ Environment validation passed")
    return True


def initialize_speech_recognition() -> sr.Recognizer:
    """
    Initializes and configures the speech recognizer.
    
    Returns:
        Configured Recognizer instance
    """
    recognizer = sr.Recognizer()
    
    # Adjust for ambient noise on first run
    recognizer.energy_threshold = 4000  # Sensitivity threshold
    recognizer.dynamic_energy_threshold = True  # Auto-adjust to ambient noise
    
    print("🎤 Speech recognition initialized")
    return recognizer


def initialize_audio_playback() -> None:
    """
    Initializes the pygame mixer for audio playback.
    """
    pygame.mixer.init()
    print("🔊 Audio playback initialized")


def initialize_vectorstore() -> Chroma:
    """
    Loads the existing ChromaDB vector store.
    
    Returns:
        Chroma vector store instance
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DB_PATH),
        embedding_function=embeddings
    )
    
    doc_count = vectorstore._collection.count()
    print(f"📚 Vector store loaded ({doc_count} documents)")
    return vectorstore


def initialize_llm() -> ChatGoogleGenerativeAI:
    """
    Initializes the Gemini LLM.
    
    Returns:
        ChatGoogleGenerativeAI instance
    """
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=TEMPERATURE,
        convert_system_message_to_human=True  # Gemini compatibility
    )
    
    print(f"🤖 Gemini LLM initialized (model: {GEMINI_MODEL})")
    return llm


# ============================================
# Core Functions
# ============================================

def listen_for_speech(recognizer: sr.Recognizer) -> str | None:
    """
    Listens to the microphone and transcribes speech to text.
    
    Args:
        recognizer: Configured Recognizer instance
        
    Returns:
        Transcribed text, or None if an error occurred
    """
    try:
        with sr.Microphone() as source:
            print("\n🎤 Listening... (speak now)")
            
            # Adjust for ambient noise briefly
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            # Listen for audio
            audio = recognizer.listen(
                source,
                timeout=LISTEN_TIMEOUT,
                phrase_time_limit=PHRASE_TIME_LIMIT
            )
            
        print("🔄 Transcribing...")
        
        # Use Google's free speech recognition
        text = recognizer.recognize_google(audio)
        print(f"📝 You said: \"{text}\"")
        return text
        
    except sr.WaitTimeoutError:
        print("⏱️  No speech detected (timeout). Try again...")
        return None
        
    except sr.UnknownValueError:
        print("❓ Could not understand the audio. Please speak clearly...")
        return None
        
    except sr.RequestError as e:
        print(f"❌ Speech recognition service error: {e}")
        return None
        
    except OSError as e:
        print(f"❌ Microphone error: {e}")
        print("   Please check that your microphone is connected and accessible")
        return None


def retrieve_context(vectorstore: Chroma, query: str) -> str:
    """
    Retrieves relevant context from the vector store.
    
    Args:
        vectorstore: Chroma vector store
        query: User's question
        
    Returns:
        Combined context from relevant documents
    """
    results = vectorstore.similarity_search(query, k=RETRIEVAL_K)
    
    if not results:
        return "No relevant information found in the knowledge base."
    
    # Combine all retrieved chunks
    context = "\n\n---\n\n".join([doc.page_content for doc in results])
    return context


def generate_response(llm: ChatGoogleGenerativeAI, context: str, question: str) -> str:
    """
    Generates a response using Gemini with RAG context.
    
    Args:
        llm: Gemini LLM instance
        context: Retrieved context from vector store
        question: User's question
        
    Returns:
        Generated response text
    """
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({
            "context": context,
            "question": question
        })
        return response
        
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return "I'm sorry, I encountered an error generating a response. Please try again."


def speak_response(text: str) -> None:
    """
    Converts text to speech and plays it.
    
    Args:
        text: Text to speak
    """
    try:
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
            temp_path = fp.name
        
        # Generate speech
        print("🔊 Speaking...")
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(temp_path)
        
        # Play audio
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
        
        # Wait for playback to complete
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        
        # Clean up
        pygame.mixer.music.unload()
        os.unlink(temp_path)
        
    except Exception as e:
        print(f"❌ Text-to-speech error: {e}")
        print(f"   Response was: {text}")


def is_exit_command(text: str) -> bool:
    """
    Checks if the user wants to exit.
    
    Args:
        text: User's transcribed speech
        
    Returns:
        True if it's an exit command
    """
    return text.lower().strip() in EXIT_COMMANDS


# ============================================
# Main Application Loop
# ============================================

def main():
    """
    Main application entry point.
    """
    print("\n" + "=" * 50)
    print("🎙️  Voice AI Assistant with RAG")
    print("=" * 50)
    print("   Say 'exit' or 'stop' to quit")
    print("=" * 50 + "\n")
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Initialize components
    print("\n🔧 Initializing components...")
    
    try:
        recognizer = initialize_speech_recognition()
        initialize_audio_playback()
        vectorstore = initialize_vectorstore()
        llm = initialize_llm()
    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")
        sys.exit(1)
    
    print("\n✅ Ready! Start speaking...\n")
    print("-" * 50)
    
    # Main interaction loop
    while True:
        try:
            # Step 1: Listen for speech
            user_input = listen_for_speech(recognizer)
            
            # Handle failed transcription
            if user_input is None:
                continue
            
            # Step 2: Check for exit command
            if is_exit_command(user_input):
                print("\n👋 Goodbye!")
                speak_response("Goodbye! Have a great day!")
                break
            
            # Step 3: Retrieve relevant context
            print("🔍 Searching knowledge base...")
            context = retrieve_context(vectorstore, user_input)
            
            # Step 4: Generate response
            print("🤖 Generating response...")
            response = generate_response(llm, context, user_input)
            print(f"💬 Response: {response}")
            
            # Step 5: Speak the response
            speak_response(response)
            
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
            
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("   Continuing to listen...\n")
            continue
    
    # Cleanup
    pygame.mixer.quit()
    print("\n🔌 Assistant shut down gracefully.")


if __name__ == "__main__":
    main()
