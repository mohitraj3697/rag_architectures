import { useState, useEffect } from 'react'
import './App.css'
import Sidebar from './components/Sidebar'
import Playground from './components/Playground'
import DiagramView from './components/DiagramView'
import ComparisonMatrix from './components/ComparisonMatrix'

const API_BASE = 'http://localhost:8000'

function App() {
  const [architectures, setArchitectures] = useState({})
  const [selectedArch, setSelectedArch] = useState('')
  const [activeTab, setActiveTab] = useState('playground')
  const [model, setModel] = useState('openai/gpt-oss-20b')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch(`${API_BASE}/api/architectures`)
      .then(res => res.json())
      .then(data => {
        setArchitectures(data)
        const keys = Object.keys(data)
        if (keys.length > 0) setSelectedArch(keys[0])
        setLoading(false)
      })
      .catch(() => {
        setError('Cannot connect to backend. Make sure the FastAPI server is running on port 8000.')
        setLoading(false)
      })
  }, [])

  if (loading) {
    return (
      <div className="app-layout">
        <div className="spinner-overlay" style={{ flex: 1 }} role="status">
          <div className="spinner" aria-hidden="true"></div>
          <span>Loading Welcome RAG Suite...</span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="app-layout">
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 40 }}>
          <div className="status-bar status-error" style={{ maxWidth: 600 }} role="alert">
            {error}
          </div>
        </div>
      </div>
    )
  }

  const archInfo = architectures[selectedArch] || {}
  const archKeys = Object.keys(architectures)

  const tabs = [
    { id: 'playground', label: 'Live Playground' },
    { id: 'diagram', label: 'Architecture Diagram and Specs' },
    { id: 'matrix', label: '10 RAG Comparison Matrix' },
  ]

  const handleTabKeyDown = (e, index) => {
    if (e.key === 'ArrowRight') {
      e.preventDefault()
      const nextIndex = (index + 1) % tabs.length
      setActiveTab(tabs[nextIndex].id)
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault()
      const prevIndex = (index - 1 + tabs.length) % tabs.length
      setActiveTab(tabs[prevIndex].id)
    }
  }

  return (
    <div className="app-layout">
      <Sidebar
        architectures={architectures}
        archKeys={archKeys}
        selectedArch={selectedArch}
        onSelectArch={setSelectedArch}
        model={model}
        onModelChange={setModel}
        apiBase={API_BASE}
      />

      <main className="main-content" id="main-content">
        {/* Header */}
        <header className="page-header">
          <h1 className="page-header-title">{archInfo.title}</h1>
          <p className="page-header-subtitle">{archInfo.subtitle}</p>
          <div className="badge-row" aria-label="Architecture attributes">
            {(archInfo.badges || []).map((b, i) => (
              <span key={i} className={`badge badge-${b.color}`}>{b.label}</span>
            ))}
          </div>
        </header>

        {/* Tabs with ARIA accessibility */}
        <div className="tabs" role="tablist" aria-label="Architecture View Navigation">
          {tabs.map((tab, idx) => (
            <button
              key={tab.id}
              id={`tab-${tab.id}`}
              role="tab"
              aria-selected={activeTab === tab.id}
              aria-controls={`panel-${tab.id}`}
              tabIndex={activeTab === tab.id ? 0 : -1}
              className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
              onKeyDown={e => handleTabKeyDown(e, idx)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div
          className="tab-content"
          id={`panel-${activeTab}`}
          role="tabpanel"
          aria-labelledby={`tab-${activeTab}`}
        >
          {activeTab === 'playground' && (
            <Playground
              archKey={selectedArch}
              archInfo={archInfo}
              apiBase={API_BASE}
            />
          )}
          {activeTab === 'diagram' && (
            <DiagramView archInfo={archInfo} />
          )}
          {activeTab === 'matrix' && (
            <ComparisonMatrix />
          )}
        </div>
      </main>
    </div>
  )
}

export default App
