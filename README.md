# 🎙️ Voice AI Assistant with RAG

A production-grade voice assistant that uses RAG (Retrieval-Augmented Generation) to answer questions from your PDF documents using Google Gemini.

## Features

- **Voice Input**: Speak naturally to the assistant
- **RAG Pipeline**: Searches your PDF documents for relevant context
- **Gemini-Powered**: Uses `gemini-1.5-flash` for fast, accurate responses
- **Voice Output**: Speaks answers back to you
- **Local Storage**: ChromaDB vector store persisted locally

## Quick Start

### 1. Setup Environment

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync
```

> **Note (Linux)**: You may need to install PortAudio for PyAudio:
> ```bash
> sudo apt-get install portaudio19-dev
> ```

### 2. Configure API Key

Get your Google API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

```bash
# Option 1: Environment variable
export GOOGLE_API_KEY='your_api_key_here'

# Option 2: Create .env file
cp .env.example .env
# Edit .env and add your key
```

### 3. Add Your PDF

Place your PDF document at:
```
data/source.pdf
```

### 4. Build Knowledge Base

```bash
python ingest.py
```

This creates embeddings from your PDF and stores them in `chroma_db/`.

### 5. Run the Assistant

**Option A: CLI Mode**
```bash
uv run python app.py
```

**Option B: Web Interface** (recommended)
```bash
# Terminal 1: Start backend
uv run uvicorn backend.main:app --reload

# Terminal 2: Start frontend
cd frontend && npm run dev
```

Then open http://localhost:5173 in Chrome/Edge.

- 🎤 Click **Start** to begin listening
- 🗣️ Speak your question
- 📝 See transcription + AI response
- 🔊 Hear the response spoken back
- Say **"exit"** or **"stop"** to quit (CLI mode)

## Project Structure

```
├── pyproject.toml     # Project config & dependencies (uv)
├── requirements.txt   # Legacy pip support
├── .env.example       # API key template
├── data/
│   └── source.pdf     # Your PDF document
├── chroma_db/         # Vector store (auto-created)
├── ingest.py          # PDF ingestion script
├── app.py             # Main voice assistant
└── README.md          # This file
```

## Configuration

Edit constants in `app.py` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GEMINI_MODEL` | `gemini-1.5-flash` | LLM model for responses |
| `TEMPERATURE` | `0.3` | Response creativity (0-1) |
| `LISTEN_TIMEOUT` | `5` | Seconds to wait for speech |
| `PHRASE_TIME_LIMIT` | `15` | Max phrase duration |
| `RETRIEVAL_K` | `3` | Number of context chunks |

## Troubleshooting

### Microphone not detected
```bash
# Linux: Check audio devices
arecord -l

# Install ALSA utils if needed
sudo apt-get install alsa-utils
```

### "Could not understand audio"
- Speak clearly and closer to the microphone
- Reduce background noise
- Check microphone permissions

### API Key errors
- Ensure `GOOGLE_API_KEY` is set correctly
- Verify the key has Gemini API access enabled

## License

MIT
