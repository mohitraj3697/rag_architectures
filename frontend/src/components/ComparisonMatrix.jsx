const MATRIX_DATA = [
  { num: '01', pattern: 'Standard RAG', multiStep: false, selfCorrect: false, memory: 'No', cost: 'Low', bestUse: 'Static FAQ, Search' },
  { num: '02', pattern: 'Agentic RAG', multiStep: true, selfCorrect: true, memory: 'No', cost: 'High', bestUse: 'Research, Financial Analysis' },
  { num: '03', pattern: 'RAG with Memory', multiStep: false, selfCorrect: false, memory: 'Short + Long', cost: 'Medium', bestUse: 'Personal Assistants, Chatbots' },
  { num: '04', pattern: 'Self RAG', multiStep: true, selfCorrect: true, memory: 'No', cost: 'High', bestUse: 'High-accuracy QA, Legal/Medical' },
  { num: '05', pattern: 'Adaptive RAG', multiStep: 'Conditional', selfCorrect: false, memory: 'No', cost: 'Adaptive', bestUse: 'Mixed simple/complex workloads' },
  { num: '06', pattern: 'Corrective RAG', multiStep: true, selfCorrect: true, memory: 'No', cost: 'Medium', bestUse: 'Noisy data, Enterprise search' },
  { num: '07', pattern: 'Attention RAG', multiStep: false, selfCorrect: false, memory: 'No', cost: 'Medium', bestUse: 'Long documents, Summarization' },
  { num: '08', pattern: 'HybridAI RAG', multiStep: true, selfCorrect: false, memory: 'No', cost: 'Medium-High', bestUse: 'Knowledge graphs, Entity relations' },
  { num: '09', pattern: 'Cost-Constrained RAG', multiStep: false, selfCorrect: false, memory: 'No', cost: 'Bounded', bestUse: 'SaaS at scale, Bounded APIs' },
  { num: '10', pattern: 'XAI RAG', multiStep: false, selfCorrect: false, memory: 'No', cost: 'Medium', bestUse: 'Auditability, Compliance, Finance' },
]

function BoolCell({ value }) {
  if (value === true) return <span className="cell-yes">Yes</span>
  if (value === false) return <span className="cell-no">No</span>
  return <span style={{ color: 'var(--color-text-inverse)' }}>{value}</span>
}

export default function ComparisonMatrix() {
  return (
    <div>
      <h2 style={{ fontSize: 'var(--font-size-lg)', fontWeight: 'var(--font-weight-semibold)', marginBottom: 'var(--space-20)', color: 'var(--color-text-secondary)' }}>
        Master Comparison Matrix -- 10 RAG Patterns
      </h2>
      <div style={{ overflowX: 'auto' }}>
        <table className="matrix-table" aria-label="10 RAG Patterns Comparison Matrix">
          <thead>
            <tr>
              <th scope="col">#</th>
              <th scope="col">Pattern</th>
              <th scope="col">Multi-Step</th>
              <th scope="col">Self-Correct</th>
              <th scope="col">Memory</th>
              <th scope="col">Cost</th>
              <th scope="col">Best Use Case</th>
            </tr>
          </thead>
          <tbody>
            {MATRIX_DATA.map(row => (
              <tr key={row.num}>
                <td style={{ fontWeight: 'var(--font-weight-semibold)', color: 'var(--color-text-inverse)' }}>{row.num}</td>
                <td style={{ fontWeight: 'var(--font-weight-medium)', color: 'var(--color-text-secondary)' }}>{row.pattern}</td>
                <td><BoolCell value={row.multiStep} /></td>
                <td><BoolCell value={row.selfCorrect} /></td>
                <td>{row.memory === 'No' ? <span className="cell-no">No</span> : <span className="cell-yes">{row.memory}</span>}</td>
                <td style={{ color: 'var(--color-text-primary)' }}>{row.cost}</td>
                <td style={{ color: 'var(--color-text-primary)' }}>{row.bestUse}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
