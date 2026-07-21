import { useState, useEffect } from 'react'
import ResultDisplay from './ResultDisplay'

export default function Playground({ archKey, archInfo, apiBase }) {
  const [query, setQuery] = useState(archInfo.default_query || '')
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [budget, setBudget] = useState(1000)
  const [userId, setUserId] = useState('user-42')

  useEffect(() => {
    setQuery(archInfo.default_query || '')
    setResult(null)
    setError(null)
  }, [archKey, archInfo.default_query])

  async function handleRun() {
    if (!query.trim()) return
    setRunning(true)
    setError(null)
    setResult(null)

    const extraParams = {}
    if (archKey === '09_cost_constrained_rag') {
      extraParams.budget_remaining = budget
    }
    if (archKey === '03_memory_rag') {
      extraParams.user_id = userId
    }

    try {
      const res = await fetch(`${apiBase}/api/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          architecture: archKey,
          query: query,
          extra_params: Object.keys(extraParams).length > 0 ? extraParams : null,
        }),
      })
      const data = await res.json()
      if (res.ok) {
        setResult(data)
      } else {
        setError(data.detail || 'An error occurred.')
      }
    } catch (e) {
      setError(`Network error: ${e.message}`)
    } finally {
      setRunning(false)
    }
  }

  return (
    <div>
      <h2 style={{ fontSize: 'var(--font-size-lg)', fontWeight: 'var(--font-weight-semibold)', marginBottom: 'var(--space-16)', color: 'var(--color-text-secondary)' }}>
        Test Architecture Execution
      </h2>

      {/* Query Input Area */}
      <div className="query-area">
        <div>
          <label htmlFor="query-input" className="form-label">
            Enter your Question
          </label>
          <textarea
            id="query-input"
            className="form-textarea query-textarea"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Type your question for the RAG pipeline..."
            rows={4}
          />
        </div>
        <div className="sample-card">
          <div className="sample-card-title">Sample Slide Query</div>
          <div className="sample-card-text">{archInfo.slide_example}</div>
        </div>
      </div>

      {/* Extra Params */}
      {archKey === '09_cost_constrained_rag' && (
        <div style={{ marginBottom: 'var(--space-16)' }}>
          <label htmlFor="budget-range" className="form-label">
            Starting Token Budget: {budget}
          </label>
          <input
            id="budget-range"
            type="range"
            min={100}
            max={2000}
            step={100}
            value={budget}
            onChange={e => setBudget(Number(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--color-text-inverse)' }}
          />
          <p style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)', marginTop: 'var(--space-4)' }}>
            Values below 400 trigger Economy tier (smaller model + fewer chunks)
          </p>
        </div>
      )}
      {archKey === '03_memory_rag' && (
        <div style={{ marginBottom: 'var(--space-16)' }}>
          <label htmlFor="user-id-input" className="form-label">
            User ID
          </label>
          <input
            id="user-id-input"
            className="form-input"
            value={userId}
            onChange={e => setUserId(e.target.value)}
            style={{ maxWidth: 300 }}
          />
        </div>
      )}

      {/* Run Button */}
      <button
        className="btn btn-primary btn-full"
        onClick={handleRun}
        disabled={running || !query.trim()}
        aria-busy={running}
      >
        {running ? 'Running Pipeline...' : 'Run Architecture Pipeline'}
      </button>

      {/* Loading State */}
      {running && (
        <div className="spinner-overlay" role="status">
          <div className="spinner" aria-hidden="true"></div>
          <span>Executing {archInfo.title} graph via Groq API...</span>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="status-bar status-error" style={{ marginTop: 'var(--space-16)' }} role="alert">
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <ResultDisplay archKey={archKey} result={result} />
      )}
    </div>
  )
}
