import { useState } from 'react';
import VoiceAssistant from './VoiceAssistant';
import AdminPanel from './AdminPanel';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('assistant');

  return (
    <div className="app">
      {/* Tab Navigation */}
      <nav className="tab-nav">
        <button
          className={`tab-button ${activeTab === 'assistant' ? 'active' : ''}`}
          onClick={() => setActiveTab('assistant')}
        >
          <span className="tab-icon">🎙️</span>
          Voice Assistant
        </button>
        <button
          className={`tab-button ${activeTab === 'admin' ? 'active' : ''}`}
          onClick={() => setActiveTab('admin')}
        >
          <span className="tab-icon">⚙️</span>
          Admin Panel
        </button>
      </nav>

      {/* Content */}
      <main className="app-content">
        {activeTab === 'assistant' ? <VoiceAssistant /> : <AdminPanel />}
      </main>
    </div>
  );
}

export default App;
