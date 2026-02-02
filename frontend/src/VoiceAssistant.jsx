import { useState, useRef, useEffect } from 'react';
import { useSpeechRecognition } from './useSpeechRecognition';
import { queryRAG, textToSpeech, checkHealth } from './api';
import './VoiceAssistant.css';

function VoiceAssistant() {
    const [messages, setMessages] = useState([]);
    const [isProcessing, setIsProcessing] = useState(false);
    const [isPlaying, setIsPlaying] = useState(false);
    const [backendStatus, setBackendStatus] = useState('checking');
    const audioRef = useRef(null);
    const messagesEndRef = useRef(null);

    const {
        isListening,
        transcript,
        error: speechError,
        isSupported,
        startListening,
        stopListening,
    } = useSpeechRecognition();

    // Check backend health on mount
    useEffect(() => {
        checkHealth()
            .then(() => setBackendStatus('connected'))
            .catch(() => setBackendStatus('disconnected'));
    }, []);

    // Auto-scroll messages
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // Process transcript when listening stops
    useEffect(() => {
        if (!isListening && transcript && !isProcessing) {
            handleQuery(transcript);
        }
    }, [isListening, transcript]);

    const handleQuery = async (question) => {
        if (!question.trim()) return;

        // Add user message
        setMessages(prev => [...prev, { role: 'user', content: question }]);
        setIsProcessing(true);

        try {
            // Query RAG backend
            const response = await queryRAG(question);

            // Add assistant message
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: response.answer,
                sources: response.sources
            }]);

            // Convert to speech and play
            const audioUrl = await textToSpeech(response.answer);
            playAudio(audioUrl);

        } catch (err) {
            setMessages(prev => [...prev, {
                role: 'error',
                content: err.message
            }]);
        } finally {
            setIsProcessing(false);
        }
    };

    const playAudio = (url) => {
        if (audioRef.current) {
            audioRef.current.src = url;
            audioRef.current.play();
            setIsPlaying(true);
        }
    };

    const handleAudioEnd = () => {
        setIsPlaying(false);
        if (audioRef.current?.src) {
            URL.revokeObjectURL(audioRef.current.src);
        }
    };

    const handleMicClick = () => {
        if (isListening) {
            stopListening();
        } else {
            startListening();
        }
    };

    const getStatusColor = () => {
        if (backendStatus === 'connected') return '#10b981';
        if (backendStatus === 'disconnected') return '#ef4444';
        return '#f59e0b';
    };

    return (
        <div className="voice-assistant">
            {/* Header */}
            <header className="header">
                <h1>🎙️ Voice AI Assistant</h1>
                <div className="status" style={{ '--status-color': getStatusColor() }}>
                    <span className="status-dot"></span>
                    Backend: {backendStatus}
                </div>
            </header>

            {/* Messages */}
            <div className="messages">
                {messages.length === 0 && (
                    <div className="empty-state">
                        <p>Click the microphone and ask a question about your documents</p>
                    </div>
                )}

                {messages.map((msg, idx) => (
                    <div key={idx} className={`message ${msg.role}`}>
                        <div className="message-content">
                            {msg.role === 'user' && <span className="icon">🧑</span>}
                            {msg.role === 'assistant' && <span className="icon">🤖</span>}
                            {msg.role === 'error' && <span className="icon">❌</span>}
                            <p>{msg.content}</p>
                        </div>
                    </div>
                ))}

                {isProcessing && (
                    <div className="message assistant">
                        <div className="message-content">
                            <span className="icon">🤖</span>
                            <div className="typing-indicator">
                                <span></span><span></span><span></span>
                            </div>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Current transcript */}
            {transcript && isListening && (
                <div className="transcript">
                    <p>"{transcript}"</p>
                </div>
            )}

            {/* Controls */}
            <div className="controls">
                {!isSupported && (
                    <p className="warning">Speech recognition not supported in this browser. Try Chrome or Edge.</p>
                )}

                {speechError && (
                    <p className="error">{speechError}</p>
                )}

                <button
                    className={`mic-button ${isListening ? 'listening' : ''} ${isProcessing ? 'processing' : ''}`}
                    onClick={handleMicClick}
                    disabled={!isSupported || isProcessing || backendStatus !== 'connected'}
                >
                    {isListening ? (
                        <>
                            <span className="pulse"></span>
                            🎤 Listening...
                        </>
                    ) : isProcessing ? (
                        '⏳ Processing...'
                    ) : isPlaying ? (
                        '🔊 Speaking...'
                    ) : (
                        '🎤 Start'
                    )}
                </button>
            </div>

            {/* Hidden audio element */}
            <audio
                ref={audioRef}
                onEnded={handleAudioEnd}
                onError={handleAudioEnd}
            />
        </div>
    );
}

export default VoiceAssistant;
