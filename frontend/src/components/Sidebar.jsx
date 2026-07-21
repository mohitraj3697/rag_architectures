import { useState, useRef } from 'react'

export default function Sidebar({ architectures, archKeys, selectedArch, onSelectArch, model, onModelChange, apiBase }) {
  const [uploading, setUploading] = useState(false)
  const [uploadMsg, setUploadMsg] = useState('')
  const fileRef = useRef(null)

  const models = [
    'openai/gpt-oss-20b',
    'openai/gpt-oss-120b',
    'llama-3.3-70b-versatile',
    'mixtral-8x7b-32768',
  ]

  async function handleUpload() {
    const files = fileRef.current?.files
    if (!files || files.length === 0) {
      setUploadMsg('Please select files first.')
      return
    }
    setUploading(true)
    setUploadMsg('')
    try {
      const formData = new FormData()
      for (const f of files) formData.append('files', f)
      const res = await fetch(`${apiBase}/api/upload`, { method: 'POST', body: formData })
      const data = await res.json()
      if (res.ok) {
        setUploadMsg(`Indexed ${data.count} document(s) successfully.`)
      } else {
        setUploadMsg(`Error: ${data.detail}`)
      }
    } catch (e) {
      setUploadMsg(`Upload failed: ${e.message}`)
    } finally {
      setUploading(false)
    }
  }

  async function handleReset() {
    try {
      await fetch(`${apiBase}/api/reset-kb`, { method: 'POST' })
      setUploadMsg('Reset to default knowledge base.')
    } catch (e) {
      setUploadMsg(`Reset failed: ${e.message}`)
    }
  }

  const handleKeyDown = (e, key) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      onSelectArch(key)
    }
  }

  return (
    <aside className="sidebar" aria-label="Architecture Navigation Sidebar">
      <div className="sidebar-header">
        <h1>Welcome RAG Suite</h1>
        <p>Token-Driven Architecture Playground</p>
      </div>

      {/* Model Selector */}
      <div className="sidebar-section">
        <label htmlFor="model-select" className="sidebar-section-title" style={{ display: 'block' }}>
          LLM Model
        </label>
        <select
          id="model-select"
          className="form-select"
          value={model}
          onChange={e => onModelChange(e.target.value)}
          aria-label="Select LLM Model"
        >
          {models.map(m => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      </div>

      {/* Architecture Selector */}
      <div className="sidebar-section" style={{ flex: 1, overflowY: 'auto' }}>
        <div className="sidebar-section-title" id="arch-group-label">Architecture</div>
        <div className="arch-radio-group" role="radiogroup" aria-labelledby="arch-group-label">
          {archKeys.map(key => {
            const isSelected = selectedArch === key
            return (
              <button
                key={key}
                type="button"
                role="radio"
                aria-checked={isSelected}
                tabIndex={0}
                className={`arch-radio-item ${isSelected ? 'active' : ''}`}
                onClick={() => onSelectArch(key)}
                onKeyDown={e => handleKeyDown(e, key)}
              >
                <div className="radio-dot" aria-hidden="true"></div>
                <span>{architectures[key].title}</span>
              </button>
            )
          })}
        </div>
      </div>

      {/* Upload Section */}
      <div className="sidebar-section">
        <div className="sidebar-section-title">Custom Knowledge Base</div>
        <div
          className="upload-zone"
          tabIndex={0}
          role="button"
          aria-label="Upload custom document files"
          onClick={() => fileRef.current?.click()}
          onKeyDown={e => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              fileRef.current?.click()
            }
          }}
        >
          <input
            ref={fileRef}
            type="file"
            accept=".txt,.md,.csv,.json,.py"
            multiple
            aria-hidden="true"
          />
          <p>Click or press Enter to select files (.txt, .md, .csv, .json)</p>
        </div>
        <button
          className="btn btn-primary btn-full"
          onClick={handleUpload}
          disabled={uploading}
          style={{ marginBottom: 8 }}
          aria-busy={uploading}
        >
          {uploading ? 'Indexing...' : 'Index in Vector DB'}
        </button>
        <button className="btn btn-secondary btn-full" onClick={handleReset}>
          Reset to Default KB
        </button>
        {uploadMsg && (
          <p
            role="status"
            aria-live="polite"
            style={{
              marginTop: 8,
              fontSize: 'var(--font-size-xs)',
              color: uploadMsg.includes('Error') || uploadMsg.includes('failed') ? 'var(--color-status-error)' : 'var(--color-status-success)'
            }}
          >
            {uploadMsg}
          </p>
        )}
      </div>
    </aside>
  )
}
