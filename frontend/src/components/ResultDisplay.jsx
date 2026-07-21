export default function ResultDisplay({ archKey, result }) {
  const { answer, elapsed, documents, traces } = result

  return (
    <div style={{ marginTop: 'var(--space-24)' }}>
      <div className="status-bar status-success" role="status" aria-live="polite">
        Execution completed in {elapsed}s
      </div>

      <div className="result-grid">
        {/* Main Answer */}
        <div className="result-card">
          <h3>Generated Response</h3>
          <div className="answer-block">{answer}</div>

          {/* Architecture-specific traces */}
          {archKey === '04_self_rag' && traces && (
            <div className="trace-section">
              <h4>Self-RAG Retries and Verdict</h4>
              <div className="trace-item">
                <span className="trace-label">Retries Used</span>
                <span className="trace-value">{traces.retries ?? 0} / 2</span>
              </div>
              <div className="trace-item">
                <span className="trace-label">Final Verdict</span>
                <span className="trace-value">{traces.verdict ?? 'N/A'}</span>
              </div>
            </div>
          )}

          {archKey === '05_adaptive_rag' && traces && (
            <div className="trace-section">
              <h4>Complexity Routing and Sub-Queries</h4>
              <div className="trace-item">
                <span className="trace-label">Routed As</span>
                <span className="trace-value">{(traces.complexity || '').toUpperCase()}</span>
              </div>
              {traces.sub_queries && traces.sub_queries.length > 0 && (
                <div style={{ marginTop: 'var(--space-8)' }}>
                  <p style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)', marginBottom: 'var(--space-4)' }}>Generated Sub-Queries:</p>
                  {traces.sub_queries.map((sq, i) => (
                    <p key={i} style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)', paddingLeft: 'var(--space-12)' }}>- {sq}</p>
                  ))}
                </div>
              )}
            </div>
          )}

          {archKey === '06_corrective_rag' && traces && (
            <div className="trace-section">
              <h4>Corrective RAG Filtering</h4>
              <div className="trace-item">
                <span className="trace-label">Relevant Document Ratio</span>
                <span className="trace-value">{(traces.relevant_ratio ?? 0).toFixed(2)}</span>
              </div>
              <div className="trace-item">
                <span className="trace-label">Re-queries Used</span>
                <span className="trace-value">{traces.retries ?? 0}</span>
              </div>
            </div>
          )}

          {archKey === '07_attention_rag' && traces?.weighted && (
            <div className="trace-section">
              <h4>Passage Attention Weights</h4>
              {traces.weighted.map((w, i) => (
                <div key={i} className="trace-item">
                  <span className="trace-label">{w.source}</span>
                  <span className="trace-value">{w.weight.toFixed(2)}</span>
                </div>
              ))}
            </div>
          )}

          {archKey === '08_hybrid_rag' && traces && (
            <div className="trace-section">
              <h4>Knowledge Graph Triples Used</h4>
              <div className="trace-item">
                <span className="trace-label">Entities Detected</span>
                <span className="trace-value">{(traces.entities || []).join(', ') || 'None'}</span>
              </div>
              {(traces.kg_facts || []).map((fact, i) => (
                <div key={i} className="trace-item">
                  <span className="trace-label">KG Fact {i + 1}</span>
                  <span className="trace-value" style={{ fontSize: 'var(--font-size-xs)' }}>{fact}</span>
                </div>
              ))}
            </div>
          )}

          {archKey === '09_cost_constrained_rag' && traces && (
            <div className="trace-section">
              <h4>Cost and Budget Tracking</h4>
              <div className="trace-item">
                <span className="trace-label">Tier Used</span>
                <span className="trace-value">{(traces.tier || '').toUpperCase()}</span>
              </div>
              <div className="trace-item">
                <span className="trace-label">Model</span>
                <span className="trace-value">{traces.model}</span>
              </div>
              <div className="trace-item">
                <span className="trace-label">Tokens Spent</span>
                <span className="trace-value">{traces.tokens_spent}</span>
              </div>
              <div className="trace-item">
                <span className="trace-label">Budget Remaining</span>
                <span className="trace-value">{traces.budget_remaining}</span>
              </div>
            </div>
          )}

          {archKey === '10_xai_rag' && traces && (
            <div className="trace-section">
              <h4>Reasoning Trace and Citations</h4>
              {traces.reasoning_trace && (
                <div className="answer-block" style={{ marginBottom: 'var(--space-12)', fontSize: 'var(--font-size-xs)', color: 'var(--color-status-success)' }}>
                  {traces.reasoning_trace}
                </div>
              )}
              <div className="trace-item">
                <span className="trace-label">Cited Source IDs</span>
                <span className="trace-value">{JSON.stringify(traces.cited_source_ids || [])}</span>
              </div>
            </div>
          )}

          {archKey === '03_memory_rag' && traces?.long_term_memory && (
            <div className="trace-section">
              <h4>Long-Term User Profile Memory</h4>
              <div className="answer-block" style={{ fontSize: 'var(--font-size-xs)' }}>
                {JSON.stringify(traces.long_term_memory, null, 2)}
              </div>
            </div>
          )}

          {archKey === '02_agentic_rag' && traces?.tool_results && traces.tool_results.length > 0 && (
            <div className="trace-section">
              <h4>Tool Call Results</h4>
              {traces.tool_results.map((tr, i) => (
                <div key={i} className="source-card">
                  <div className="source-card-title">Tool Result {i + 1}</div>
                  <div className="source-card-text">{tr}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Sources Column */}
        <div className="result-card">
          <h3>Context and Sources</h3>
          {documents && documents.length > 0 ? (
            documents.map((doc, i) => (
              <div key={i} className="source-card">
                <div className="source-card-title">[{i + 1}] {doc.metadata?.source || 'Unknown'}</div>
                <div className="source-card-text">{doc.page_content}</div>
              </div>
            ))
          ) : (
            <p style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)' }}>
              No direct vector documents retrieved (e.g. direct answer or tool-based).
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
