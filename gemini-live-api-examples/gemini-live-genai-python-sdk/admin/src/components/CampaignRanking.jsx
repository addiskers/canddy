import { useCallback, useEffect, useState } from 'react'
import { api, getToken } from '../api.js'
import { fmtDate, ScoreBadge } from './CallLogs.jsx'
import { IconDownload } from './icons.jsx'

// Per-employee comparison for a campaign (GET /campaigns/{id}/ranking): one row
// per recipient, already sorted worst-first by the API (red-flag severity, then
// ascending score). Unassessed recipients trail with their pipeline status.
// Row click opens the underlying call in the shared drawer via onOpenCall.

function FlagPill({ level }) {
  if (!level) return <span className="muted">—</span>
  const cls = level === 'Critical' ? 'red' : level === 'Moderate' ? 'amber' : 'green'
  return <span className={`pill ${cls}`}>{level}</span>
}

export default function CampaignRanking({ campaignId, onOpenCall, onStats }) {
  const [items, setItems] = useState([])
  const [assessed, setAssessed] = useState(0)
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState('')

  const load = useCallback(() => {
    setLoading(true)
    api.get(`/campaigns/${campaignId}/ranking`)
      .then((d) => {
        const rows = d.items || []
        setItems(rows)
        setAssessed(d.assessed || 0)
        setErr('')
        if (onStats) {
          const scored = rows.filter((r) => r.total_score != null)
          onStats({
            avgScore: scored.length
              ? Math.round((scored.reduce((sum, r) => sum + r.total_score, 0) / scored.length) * 10) / 10
              : null,
          })
        }
      })
      .catch((e) => setErr(e.message))
      .finally(() => setLoading(false))
  }, [campaignId, onStats])

  useEffect(() => { load() }, [load])

  // Token in the query — plain <a> downloads can't send an Authorization header
  // (same pattern as the recording <audio> links).
  const csvHref = `/api/eo/campaigns/${campaignId}/ranking?format=csv&token=${encodeURIComponent(getToken())}`

  return (
    <div className="panel">
      <div className="panel-head">
        <h3>Employee Ranking</h3>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <span className="muted" style={{ fontSize: '0.74rem' }}>{assessed} assessed · most concerning first</span>
          <button className="btn ghost sm" onClick={load}>Refresh</button>
          <a className="btn ghost sm" href={csvHref} download style={{ display: 'inline-flex', gap: 6 }}>
            <IconDownload /> Export CSV
          </a>
        </div>
      </div>

      {err && <div style={{ color: '#fca5a5', fontSize: '0.82rem', marginBottom: 10 }}>{err}</div>}

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th className="no-sort num">Rank</th>
              <th className="no-sort">Employee</th>
              <th className="no-sort">Phone</th>
              <th className="no-sort">Score</th>
              <th className="no-sort">Red flag</th>
              <th className="no-sort">Review status</th>
              <th className="no-sort">Outcome</th>
              <th className="no-sort">Interviewed</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={8} className="empty">Loading…</td></tr>
            ) : items.length === 0 ? (
              <tr><td colSpan={8} className="empty">No recipients in this campaign.</td></tr>
            ) : items.map((r) => {
              const isAssessed = r.total_score != null
              const clickable = Boolean(r.call_id && onOpenCall)
              return (
                <tr key={r.cc_id ?? `${r.rank}-${r.phone}`} className={clickable ? 'clickable' : ''}
                    onClick={clickable ? () => onOpenCall(r.call_id) : undefined}
                    title={clickable ? 'View this employee’s interview' : ''}>
                  <td className="num">{r.rank}</td>
                  <td>
                    {r.name || <span className="muted">—</span>}
                    {r.employee_id && <span className="muted" style={{ fontSize: '0.7rem' }}> · {r.employee_id}</span>}
                  </td>
                  <td style={{ fontFamily: 'var(--mono)' }}>{r.phone}</td>
                  <td>
                    {isAssessed
                      ? <ScoreBadge score={r.total_score} redFlag={r.red_flag_level} />
                      : <span className="muted">{r.assessment_status || 'unreached'}</span>}
                  </td>
                  <td>{isAssessed ? <FlagPill level={r.red_flag_level} /> : <span className="muted">—</span>}</td>
                  <td>
                    {isAssessed ? (r.review_status || <span className="muted">—</span>) : <span className="muted">—</span>}
                    {r.overridden && <span className="muted" style={{ fontSize: '0.68rem' }}> (edited)</span>}
                  </td>
                  <td>{r.outcome_label || r.outcome || <span className="muted">—</span>}</td>
                  <td>{r.interviewed_at ? fmtDate(r.interviewed_at) : <span className="muted">—</span>}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <div className="pager"><span>{items.length} recipients</span></div>
    </div>
  )
}
