import { useState, useEffect, useRef } from 'react';
import { uploadPDF, getDocuments, checkHealth } from './api';
import './AdminPanel.css';

function AdminPanel() {
    const [documents, setDocuments] = useState([]);
    const [vectorstoreChunks, setVectorstoreChunks] = useState(0);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(null);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(null);
    const [backendStatus, setBackendStatus] = useState('checking');
    const [isDragging, setIsDragging] = useState(false);
    const fileInputRef = useRef(null);

    // Check backend health and load documents on mount
    useEffect(() => {
        checkHealth()
            .then(() => {
                setBackendStatus('connected');
                loadDocuments();
            })
            .catch(() => setBackendStatus('disconnected'));
    }, []);

    const loadDocuments = async () => {
        try {
            const data = await getDocuments();
            setDocuments(data.documents);
            setVectorstoreChunks(data.vectorstore_chunks);
        } catch (err) {
            console.error('Failed to load documents:', err);
        }
    };

    const handleFileUpload = async (file) => {
        if (!file) return;

        if (!file.name.toLowerCase().endsWith('.txt')) {
            setError('Only text (.txt) files are allowed');
            return;
        }

        setIsUploading(true);
        setError(null);
        setSuccess(null);
        setUploadProgress({ filename: file.name, status: 'uploading' });

        try {
            const result = await uploadPDF(file);
            setUploadProgress({
                filename: file.name,
                status: 'complete',
                characters: result.characters,
                chunks: result.chunks
            });
            setSuccess(`Successfully uploaded "${file.name}" (${result.characters} chars, ${result.chunks} chunks)`);
            loadDocuments();
        } catch (err) {
            setError(err.message);
            setUploadProgress(null);
        } finally {
            setIsUploading(false);
        }
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        setIsDragging(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files[0];
        handleFileUpload(file);
    };

    const handleFileSelect = (e) => {
        const file = e.target.files[0];
        handleFileUpload(file);
    };

    const formatFileSize = (bytes) => {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
    };

    const formatDate = (isoString) => {
        return new Date(isoString).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    const getStatusColor = () => {
        if (backendStatus === 'connected') return '#10b981';
        if (backendStatus === 'disconnected') return '#ef4444';
        return '#f59e0b';
    };

    return (
        <div className="admin-panel">
            {/* Header */}
            <header className="header">
                <h1>⚙️ Admin Panel</h1>
                <div className="status" style={{ '--status-color': getStatusColor() }}>
                    <span className="status-dot"></span>
                    Backend: {backendStatus}
                </div>
            </header>

            <div className="admin-content">
                {/* Stats Cards */}
                <div className="stats-grid">
                    <div className="stat-card">
                        <span className="stat-icon">📝</span>
                        <div className="stat-info">
                            <span className="stat-value">{documents.length}</span>
                            <span className="stat-label">Text Files</span>
                        </div>
                    </div>
                    <div className="stat-card">
                        <span className="stat-icon">🧩</span>
                        <div className="stat-info">
                            <span className="stat-value">{vectorstoreChunks}</span>
                            <span className="stat-label">Vector Chunks</span>
                        </div>
                    </div>
                </div>

                {/* Upload Zone */}
                <div
                    className={`upload-zone ${isDragging ? 'dragging' : ''} ${isUploading ? 'uploading' : ''}`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => !isUploading && fileInputRef.current?.click()}
                >
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept=".txt"
                        onChange={handleFileSelect}
                        style={{ display: 'none' }}
                    />

                    {isUploading ? (
                        <div className="upload-progress">
                            <div className="spinner"></div>
                            <p>Processing "{uploadProgress?.filename}"...</p>
                            <span className="upload-hint">Creating embeddings and indexing...</span>
                        </div>
                    ) : (
                        <>
                            <div className="upload-icon">📤</div>
                            <p>Drag & drop a text file here</p>
                            <span className="upload-hint">or click to browse</span>
                        </>
                    )}
                </div>

                {/* Messages */}
                {error && (
                    <div className="message error-message">
                        <span>❌</span> {error}
                    </div>
                )}

                {success && (
                    <div className="message success-message">
                        <span>✅</span> {success}
                    </div>
                )}

                {/* Documents List */}
                <div className="documents-section">
                    <h2>📚 Knowledge Base Documents</h2>

                    {documents.length === 0 ? (
                        <div className="empty-state">
                            <p>No documents uploaded yet.</p>
                            <p>Upload a text file to build your knowledge base.</p>
                        </div>
                    ) : (
                        <div className="documents-list">
                            {documents.map((doc, idx) => (
                                <div key={idx} className="document-card">
                                    <div className="document-icon">📝</div>
                                    <div className="document-info">
                                        <span className="document-name">{doc.filename}</span>
                                        <span className="document-meta">
                                            {formatFileSize(doc.size_bytes)} • {formatDate(doc.uploaded_at)}
                                        </span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

export default AdminPanel;
