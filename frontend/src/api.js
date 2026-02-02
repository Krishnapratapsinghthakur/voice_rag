const API_BASE_URL = 'http://localhost:8000';

/**
 * Query the RAG pipeline
 * @param {string} question - User's question
 * @returns {Promise<{question: string, answer: string, sources: string[]}>}
 */
export async function queryRAG(question) {
    const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Query failed');
    }

    return response.json();
}

/**
 * Convert text to speech and return audio URL
 * @param {string} text - Text to convert
 * @returns {Promise<string>} - Blob URL for audio
 */
export async function textToSpeech(text) {
    const response = await fetch(`${API_BASE_URL}/tts`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
    });

    if (!response.ok) {
        throw new Error('TTS failed');
    }

    const blob = await response.blob();
    return URL.createObjectURL(blob);
}

/**
 * Check backend health
 * @returns {Promise<{status: string}>}
 */
export async function checkHealth() {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.json();
}
