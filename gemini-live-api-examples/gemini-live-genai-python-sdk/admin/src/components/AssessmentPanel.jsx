import { useState } from 'react'
import { api } from '../api.js'

// Post-call interview assessment card for the call drawer. Renders the scored
// rubric (six category meters), classification chips, the management review
// routing (editable review status via PATCH /calls/{id}/review), evidence
// quotes, named persons, narrative summary and question coverage. Every
// assessment is human-reviewed — the amber pill is always shown.

const CATS = [
  ['involvement', 'Involvement'],
  ['conduct', 'Conduct'],
  ['accountability', 'Accountability'],
  ['future_compliance', 'Future compliance'],
  ['communication', 'Communication'],
  ['overall_suitability', 'Overall suitability'],
]

const REVIEW_STATUSES = ['Pass', 'Moderate', 'Needs review']

const scoreColor = (pct) => (pct >= 75 ? 'var(--ok-2)' : pct >= 50 ? 'var(--amber)' : '#fca5a5')
const prettyBand = (b) => String(b || '').replace(/_/g, ' ')

export default function AssessmentPanel({ call, onReload }) {
  const a = call.assessment || {}
  const id = call.id || call.call_sid
  const cls = a.classifications || {}
  const ov = a.override || {}
  const effFlag = ov.red_flag_level || cls.red_flag_level || 'None'
  const effStatus = ov.review_status || cls.review_status || ''

  // re-run (POST /calls/{id}/assess) state
  const [rerunBusy, setRerunBusy] = useState(false)
  const [rerunErr, setRerunErr] = useState('')
  // review-status editor state (same save/err/busy pattern as RsvpEditor)
  const [rs, setRs] = useState(effStatus)
  const [rsBusy, setRsBusy] = useState(false)
  const [rsErr, setRsErr] = useState('')
  const [editedBy, setEditedBy] = useState(ov.edited_by || null)

  async function rerun() {
    setRerunBusy(true); setRerunErr('')
    try {
      await api.post(`/calls/${encodeURIComponent(id)}/assess`)
      if (onReload) onReload()
    } catch (e) { setRerunErr(e.message) } finally { setRerunBusy(false) }
  }

  async function saveReview() {
    if (!rs) return
    setRsBusy(true); setRsErr('')
    try {
      const r = await api.patch(`/calls/${encodeURIComponent(id)}/review`, { review_status: rs })
      setEditedBy(r.edited_by || 'you')
    } catch (e) { setRsErr(e.message) } finally { setRsBusy(false) }
  }

  const rerunButton = (
    <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
      <button className="btn ghost sm" disabled={rerunBusy} onClick={rerun}>
        {rerunBusy ? 'Queuing…' : 'Re-run assessment'}
      </button>
      {rerunErr && <span style={{ color: '#fca5a5', fontSize: '0.72rem' }}>{rerunErr}</span>}
    </div>
  )

  if (a.status === 'pending') {
    return (
      <div className="card" style={{ marginBottom: 14, padding: 12 }}>
        <label>Assessment</label>
        <div className="muted" style={{ fontSize: '0.82rem' }}>
          <span className="live-dot" style={{ marginRight: 8 }} />Assessment running…
        </div>
      </div>
    )
  }
  if (a.status === 'failed') {
    return (
      <div className="card" style={{ marginBottom: 14, padding: 12 }}>
        <label>Assessment</label>
        <div style={{ color: '#fca5a5', fontSize: '0.8rem', marginBottom: 10 }}>
          Assessment failed{a.error ? ` — ${a.error}` : ''}
        </div>
        {rerunButton}
      </div>
    )
  }
  if (a.status === 'skipped') {
    return (
      <div className="card" style={{ marginBottom: 14, padding: 12 }}>
        <label>Assessment</label>
        <div className="muted" style={{ fontSize: '0.8rem', marginBottom: 10 }}>
          Assessment skipped{a.error ? ` — ${a.error}` : ''}
        </div>
        {rerunButton}
      </div>
    )
  }
  if (a.status !== 'completed') return null

  const total = a.total_score ?? 0
  const covered = new Set((a.question_coverage || [])
    .filter((x) => x.status === 'answered' || x.status === 'partial')
    .map((x) => x.q))
  const chips = [
    ['Involvement', cls.incident_involvement],
    ['Conduct', cls.conduct],
    ['Accountability', cls.accountability],
    ['Future compliance', cls.future_compliance],
    ['Communication', cls.communication],
    ['Red flag', effFlag],
  ].filter(([, v]) => v)

  return (
    <div className="card" style={{ marginBottom: 14, padding: 12 }}>
      <label>Assessment</label>

      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', marginBottom: 10 }}>
        <span style={{
          fontSize: '1.6rem', fontWeight: 700, lineHeight: 1,
          fontVariantNumeric: 'tabular-nums', color: scoreColor(total),
        }}>
          {total}/100
        </span>
        <span className="pill amber">Human review required</span>
      </div>

      {effFlag !== 'None' && (
        <div className={`flag-banner ${effFlag === 'Critical' ? 'critical' : 'moderate'}`} style={{ marginBottom: 12 }}>
          {effFlag === 'Critical'
            ? 'Critical red flag — review before any employment action'
            : 'Moderate red flag — management review advised'}
        </div>
      )}

      <div className="grid" style={{ gap: 10, marginBottom: 14 }}>
        {CATS.map(([key, label]) => {
          const s = (a.scores || {})[key] || {}
          const score = s.score ?? 0
          const max = s.max || 1
          const pct = Math.round((score / max) * 100)
          return (
            <div key={key}>
              <div className="row-between" style={{ marginBottom: 4 }}>
                <span style={{ fontSize: '0.76rem', color: 'var(--text-2)' }}>{label}</span>
                <span style={{ fontSize: '0.72rem', color: 'var(--secondary)', display: 'inline-flex', gap: 6, alignItems: 'center' }}>
                  <span style={{ fontFamily: 'var(--mono)' }}>{score}/{max}</span>
                  {s.band && <span className="pill src">{prettyBand(s.band)}</span>}
                </span>
              </div>
              <div className="meter">
                <div className="meter-fill" style={{ width: `${pct}%`, background: scoreColor(pct) }} />
              </div>
            </div>
          )
        })}
      </div>

      <div style={{ marginBottom: 12 }}>
        <span className="muted" style={{ fontSize: '0.7rem' }}>Classifications{a.override ? ' (management override applied)' : ''}</span>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 5 }}>
          {chips.map(([k, v]) => <span key={k} className="pill src">{k}: {v}</span>)}
        </div>
      </div>

      <div style={{ marginBottom: 12 }}>
        <span className="muted" style={{ fontSize: '0.7rem' }}>Review status</span>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginTop: 5 }}>
          <select value={rs} onChange={(e) => setRs(e.target.value)}>
            <option value="" disabled>Pick status…</option>
            {REVIEW_STATUSES.map((st) => <option key={st} value={st}>{st}</option>)}
          </select>
          <button className="btn sm" disabled={rsBusy || !rs} onClick={saveReview}>{rsBusy ? '…' : 'Save'}</button>
        </div>
        {rsErr && <div style={{ color: '#fca5a5', fontSize: '0.72rem', marginTop: 4 }}>{rsErr}</div>}
        {editedBy && <div className="muted" style={{ fontSize: '0.7rem', marginTop: 2 }}>edited by {editedBy}</div>}
      </div>

      {(a.key_evidence || []).length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <span className="muted" style={{ fontSize: '0.7rem' }}>Key evidence</span>
          {a.key_evidence.map((ev, i) => (
            <blockquote key={i} style={{
              margin: '8px 0 0', padding: '6px 10px', fontSize: '0.8rem',
              borderLeft: '3px solid var(--accent-line)',
            }}>
              <span>“{ev.quote}”</span>
              {ev.verified === false && <span className="pill red" style={{ marginLeft: 6 }}>unverified</span>}
              {ev.translation && <div className="muted" style={{ fontSize: '0.74rem', marginTop: 2 }}>{ev.translation}</div>}
              <div className="muted" style={{ fontSize: '0.7rem', marginTop: 2 }}>
                turn {ev.turn}{ev.relevance ? ` · ${ev.relevance}` : ''}
              </div>
            </blockquote>
          ))}
        </div>
      )}

      {(a.named_persons || []).length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <span className="muted" style={{ fontSize: '0.7rem' }}>Persons named by the interviewee — unverified allegations</span>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 5 }}>
            {a.named_persons.map((p, i) => (
              <span key={i} className="pill amber" title={p.alleged_conduct || ''}>{p.name}</span>
            ))}
          </div>
        </div>
      )}

      {a.summary && (
        <div style={{ marginBottom: 12 }}>
          <span className="muted" style={{ fontSize: '0.7rem' }}>Summary</span>
          <p style={{ margin: '5px 0 0', fontSize: '0.8rem', lineHeight: 1.6, color: 'var(--text-2)' }}>{a.summary}</p>
        </div>
      )}

      <div style={{ marginBottom: 12 }}>
        <span className="muted" style={{ fontSize: '0.7rem' }}>Question coverage</span>
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', alignItems: 'center', marginTop: 6 }}>
          {Array.from({ length: a.questions_total || 10 }, (_, i) => i + 1).map((n) => (
            <span key={n} title={`Q${n}`} style={{
              width: 10, height: 10, borderRadius: '50%', display: 'inline-block',
              background: covered.has(n) ? 'var(--accent)' : 'transparent',
              border: `1.5px solid ${covered.has(n) ? 'var(--accent)' : 'var(--border-strong)'}`,
            }} />
          ))}
          <span className="muted" style={{ fontSize: '0.72rem', marginLeft: 6 }}>{covered.size}/{a.questions_total || 10} questions covered</span>
        </div>
      </div>

      {rerunButton}
    </div>
  )
}
