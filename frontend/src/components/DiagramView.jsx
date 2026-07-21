import { useEffect, useRef, useState } from 'react'
import mermaid from 'mermaid'

mermaid.initialize({
  startOnLoad: false,
  theme: 'dark',
  securityLevel: 'loose',
  themeVariables: {
    darkMode: true,
    background: '#030712',
    primaryColor: '#0b0f19',
    primaryTextColor: '#ffffff',
    primaryBorderColor: 'oklch(0.75 0.183 55.934)',
    lineColor: '#374151',
    secondaryColor: '#000000',
    tertiaryColor: '#1f2937',
    fontFamily: 'Inter, sans-serif',
    fontSize: '14px',
  },
  flowchart: {
    htmlLabels: true,
    curve: 'basis',
    useMaxWidth: true,
  },
})

export default function DiagramView({ archInfo }) {
  const containerRef = useRef(null)
  const [renderError, setRenderError] = useState(null)

  useEffect(() => {
    let isMounted = true
    if (!archInfo?.mermaid || !containerRef.current) return

    const renderDiagram = async () => {
      try {
        setRenderError(null)
        const code = archInfo.mermaid.trim()
        const uniqueId = `mermaid-${Math.random().toString(36).substring(2, 9)}`

        const existing = document.getElementById(uniqueId)
        if (existing) existing.remove()

        const { svg } = await mermaid.render(uniqueId, code)
        
        if (isMounted && containerRef.current) {
          containerRef.current.innerHTML = svg
        }
      } catch (err) {
        if (isMounted && containerRef.current) {
          console.error('Mermaid render error:', err)
          setRenderError(err.message || 'Failed to render diagram')
          containerRef.current.innerHTML = `<pre style="color: var(--color-text-tertiary); font-family: monospace; font-size: var(--font-size-xs); text-align: left; background: var(--color-surface-base); padding: 16px; border-radius: var(--radius-xs); overflow-x: auto;">${archInfo.mermaid}</pre>`
        }
      }
    }

    renderDiagram()

    return () => {
      isMounted = false
    }
  }, [archInfo?.mermaid, archInfo?.title])

  return (
    <div>
      <h2 style={{ fontSize: 'var(--font-size-lg)', fontWeight: 'var(--font-weight-semibold)', marginBottom: 'var(--space-20)', color: 'var(--color-text-secondary)' }}>
        Visual Pipeline: {archInfo.title}
      </h2>

      {/* Mermaid Diagram */}
      <div className="diagram-container" ref={containerRef}>
        <div className="spinner-overlay" role="status">
          <div className="spinner" aria-hidden="true"></div>
          <span>Rendering architecture diagram...</span>
        </div>
      </div>

      {renderError && (
        <p style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)', marginBottom: 'var(--space-16)' }}>
          (Showing raw structure fallback)
        </p>
      )}

      {/* Specs Grid */}
      <div className="specs-grid">
        <div className="spec-card">
          <h4>How It Works Step-by-Step</h4>
          <ol className="step-list">
            {(archInfo.how_it_works || []).map((step, i) => (
              <li key={i} data-step={i + 1}>{step}</li>
            ))}
          </ol>
        </div>

        <div className="spec-card">
          <h4>Best When and Usage</h4>
          <div className="info-box">
            <strong>Best When:</strong> {archInfo.best_when}
          </div>
          <div className="info-box">
            <strong>Typical Use Cases:</strong> {archInfo.usage}
          </div>
        </div>
      </div>
    </div>
  )
}
