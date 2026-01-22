import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { acknowledgeAlert, fetchAlerts, fetchLocations } from '../auth/api.js'
import Map from '../components/Map.jsx'
import { clearSession, getSession } from '../auth/session.js'
import ThemeToggle from '../components/ThemeToggle.jsx'

function VoiceControls({ alert }) {
  const [lang, setLang] = useState('en-US')
  
  function speak() {
    if (!window.speechSynthesis) return
    window.speechSynthesis.cancel()
    
    const riskLevel = String(alert.risk_level || 'UNKNOWN')
    const utterance = new SpeechSynthesisUtterance()
    utterance.lang = lang
    utterance.rate = 0.9
    utterance.pitch = 1.0
    
    let text = ''
    if (lang.startsWith('hi')) {
      text = `ध्यान दें। ${riskLevel === 'HIGH' ? 'उच्च' : riskLevel === 'MEDIUM' ? 'मध्यम' : 'कम'} जोखिम चेतावनी। `
      text += `स्थान: ${alert.location || 'अज्ञात'}। `
      text += `मुख्य कारण: ${alert.primary_cause || alert.cause || 'निर्दिष्ट नहीं'}। `
      if (alert.supporting_factors && alert.supporting_factors.length > 0) {
        text += `सहायक कारक: ${alert.supporting_factors.join(', ')}। `
      }
    } else {
      text = `Attention. ${riskLevel} risk alert. `
      text += `Location: ${alert.location || 'Unknown'}. `
      text += `Primary cause: ${alert.primary_cause || alert.cause || 'Not specified'}. `
      if (alert.supporting_factors && alert.supporting_factors.length > 0) {
        text += `Supporting factors: ${alert.supporting_factors.join(', ')}. `
      }
    }
    
    utterance.text = text
    window.speechSynthesis.speak(utterance)
  }
  
  return (
    <div className="flex items-center gap-2">
      <select 
        value={lang} 
        onChange={(e) => setLang(e.target.value)}
        className="rounded-xl border border-slate-200 bg-white/70 px-2 py-1.5 text-xs font-semibold text-slate-800 hover:bg-white dark:border-white/10 dark:bg-white/10 dark:text-slate-100 dark:hover:bg-white/15"
      >
        <option value="en-US">🇬🇧 English</option>
        <option value="hi-IN">🇮🇳 हिंदी</option>
      </select>
      <button 
        type="button" 
        onClick={speak}
        className="rounded-xl border border-slate-200 bg-white/70 px-3 py-1.5 text-xs font-semibold text-slate-800 hover:bg-white dark:border-white/10 dark:bg-white/10 dark:text-slate-100 dark:hover:bg-white/15"
        title="Speak alert"
      >
        🔊 Speak
      </button>
    </div>
  )
}

export default function PoliceDashboard() {
  const nav = useNavigate()
  const session = getSession()
  const [alerts, setAlerts] = useState([])
  const [busyId, setBusyId] = useState('')
  const [error, setError] = useState('')
  const [includeAck, setIncludeAck] = useState(true)
  const [locations, setLocations] = useState([])
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [lastRefreshAt, setLastRefreshAt] = useState(null)
  const [locationQuery, setLocationQuery] = useState('')
  const [selectedUserEmail, setSelectedUserEmail] = useState('')

  function riskPill(level) {
    const v = String(level || 'NONE')
    const base = 'inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold ring-1'
    if (v === 'HIGH') return <span className={`${base} bg-red-500/10 text-red-700 ring-red-500/20 dark:text-red-300`}>HIGH</span>
    if (v === 'MEDIUM') return <span className={`${base} bg-amber-500/10 text-amber-700 ring-amber-500/20 dark:text-amber-300`}>MEDIUM</span>
    if (v === 'LOW') return <span className={`${base} bg-emerald-500/10 text-emerald-700 ring-emerald-500/20 dark:text-emerald-300`}>LOW</span>
    return <span className={`${base} bg-slate-500/10 text-slate-700 ring-slate-500/20 dark:text-slate-300`}>NONE</span>
  }

  function fmtTime(sec) {
    const s = Number(sec || 0)
    const mm = String(Math.floor(s / 60)).padStart(2, '0')
    const ss = String(Math.floor(s % 60)).padStart(2, '0')
    return `${mm}:${ss}`
  }

  function fmtScore(score) {
    const v = Number(score || 0)
    // New risk engine outputs 0..100; legacy outputs were tiny float losses/z.
    if (v > 1) return v.toFixed(0)
    return v.toFixed(6)
  }

  function fmtConfidence(c) {
    const v = Number(c)
    if (!Number.isFinite(v) || v <= 0) return '—'
    return v.toFixed(2)
  }

  const policeLabel = useMemo(() => {
    if (!session) return ''
    return `${session.email} (${session.policeId})`
  }, [session])

  const stats = useMemo(() => {
    const total = alerts.length
    const unacked = alerts.filter((a) => !a.acknowledged_at).length
    const high = alerts.filter((a) => String(a.risk_level) === 'HIGH').length
    const med = alerts.filter((a) => String(a.risk_level) === 'MEDIUM').length
    return { total, unacked, high, med }
  }, [alerts])

  const filteredLocations = useMemo(() => {
    const q = (locationQuery || '').trim().toLowerCase()
    const arr = Array.isArray(locations) ? locations : []
    if (!q) return arr
    return arr.filter((l) => {
      const email = String(l?.user_email || '').toLowerCase()
      return email.includes(q)
    })
  }, [locations, locationQuery])

  const lastRefreshLabel = useMemo(() => {
    if (!lastRefreshAt) return '—'
    try {
      return new Date(lastRefreshAt).toLocaleTimeString()
    } catch {
      return '—'
    }
  }, [lastRefreshAt])

  async function load() {
    setError('')
    try {
      setIsRefreshing(true)
      const data = await fetchAlerts({ includeAcknowledged: includeAck })
      setAlerts(data.alerts || [])
      
      const locData = await fetchLocations()
      setLocations(locData.locations || [])
      setLastRefreshAt(Date.now())
    } catch (err) {
      setError(err?.message || String(err))
    } finally {
      setIsRefreshing(false)
    }
  }

  useEffect(() => {
    load()
    const id = setInterval(load, 3000)
    return () => clearInterval(id)
  }, [includeAck])

  async function onAck(alertId) {
    setBusyId(alertId)
    try {
      await acknowledgeAlert(alertId)
      await load()
    } catch (err) {
      setError(err?.message || String(err))
    } finally {
      setBusyId('')
    }
  }

  function logout() {
    clearSession()
    nav('/')
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-sky-100 via-slate-50 to-white text-slate-900 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950 dark:text-slate-100">
      <div className="sticky top-0 z-30 border-b border-slate-200/70 bg-white/70 backdrop-blur dark:border-white/10 dark:bg-slate-950/40">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-3 sm:px-4 py-2 sm:py-3">
          <div>
            <div className="text-xs sm:text-sm font-semibold">Police Dashboard</div>
            <div className="text-[10px] sm:text-xs text-slate-600 dark:text-slate-300">Logged in as: {policeLabel}</div>
          </div>
          <div className="flex items-center gap-1 sm:gap-2">
            <ThemeToggle />
            <button onClick={logout} type="button" className="rounded-xl border border-slate-200 bg-white/70 px-2 sm:px-3 py-1.5 sm:py-2 text-xs sm:text-sm font-semibold text-slate-800 hover:bg-white dark:border-white/10 dark:bg-white/10 dark:text-slate-100 dark:hover:bg-white/15">Logout</button>
          </div>
        </div>
      </div>
      <div className="mx-auto max-w-6xl px-3 sm:px-4 py-4 sm:py-6">
        <div className="rounded-2xl border border-slate-200/70 bg-white/70 p-4 sm:p-6 shadow-sm backdrop-blur dark:border-white/10 dark:bg-white/5">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <h2 className="text-base sm:text-lg font-bold">Alerts</h2>
              <p className="mt-1 text-xs sm:text-sm text-slate-600 dark:text-slate-300">Crowd risk escalation notifications (MEDIUM/HIGH) • Polling every 3s • Persisted on disk</p>
            </div>
            <div className="flex flex-wrap items-center gap-2 sm:gap-3">
              <label className="flex items-center gap-2 text-xs sm:text-sm text-slate-700 dark:text-slate-200"><input type="checkbox" checked={includeAck} onChange={(e) => setIncludeAck(e.target.checked)} />Include acknowledged</label>
              <button onClick={load} type="button" className="rounded-xl border border-slate-200 bg-white/70 px-2 sm:px-3 py-1.5 sm:py-2 text-xs sm:text-sm font-semibold text-slate-800 hover:bg-white dark:border-white/10 dark:bg-white/10 dark:text-slate-100 dark:hover:bg-white/15">Refresh</button>
            </div>
          </div>
          <div className="mt-4 grid gap-3 grid-cols-1 md:grid-cols-2">
            <div className="rounded-2xl border border-slate-200/70 bg-white/60 p-4 text-sm text-slate-700 dark:border-white/10 dark:bg-white/5 dark:text-slate-200">
              <div className="text-xs font-semibold text-slate-600 dark:text-slate-300">Background</div>
              <div className="mt-2">Manual CCTV monitoring is slow and hard to scale. Automated, logic-driven detection helps reduce response time during crowd incidents.</div>
            </div>
            <div className="rounded-2xl border border-slate-200/70 bg-white/60 p-4 text-sm text-slate-700 dark:border-white/10 dark:bg-white/5 dark:text-slate-200">
              <div className="text-xs font-semibold text-slate-600 dark:text-slate-300">Operational objective</div>
              <div className="mt-2">Prioritize and acknowledge alerts quickly using event time, risk score, cause, and location context.</div>
            </div>
          </div>

          <div className="mt-6 rounded-2xl border border-slate-200/70 bg-white/60 p-4 dark:border-white/10 dark:bg-white/5">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h3 className="text-base font-bold">Live User Locations</h3>
                <div className="mt-1 text-xs text-slate-600 dark:text-slate-400">
                  Active users: <span className="font-semibold">{locations.length}</span>
                  <span className="text-slate-400"> • </span>
                  Last refresh: <span className="font-semibold">{lastRefreshLabel}</span>
                  <span className="text-slate-400"> • </span>
                  Status:{' '}
                  <span className={isRefreshing ? 'font-semibold text-amber-700 dark:text-amber-300' : 'font-semibold text-emerald-700 dark:text-emerald-300'}>
                    {isRefreshing ? 'Updating…' : 'Live'}
                  </span>
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <input
                  value={locationQuery}
                  onChange={(e) => setLocationQuery(e.target.value)}
                  placeholder="Search user email…"
                  className="w-full rounded-xl border border-slate-200 bg-white/70 px-3 py-2 text-sm text-slate-800 shadow-sm outline-none ring-0 placeholder:text-slate-400 focus:border-indigo-300 focus:outline-none focus:ring-2 focus:ring-indigo-200 dark:border-white/10 dark:bg-white/10 dark:text-slate-100 dark:placeholder:text-slate-400 dark:focus:border-indigo-500/50 dark:focus:ring-indigo-500/20 sm:w-64"
                />
                <button
                  type="button"
                  onClick={() => setSelectedUserEmail('')}
                  className="rounded-xl border border-slate-200 bg-white/70 px-3 py-2 text-sm font-semibold text-slate-800 hover:bg-white dark:border-white/10 dark:bg-white/10 dark:text-slate-100 dark:hover:bg-white/15"
                >
                  Clear focus
                </button>
              </div>
            </div>

            <div className="mt-4 grid gap-4 lg:grid-cols-[1fr_320px]">
              <div>
                <Map
                  locations={filteredLocations}
                  selectedUserEmail={selectedUserEmail}
                  onSelectUser={(email) => setSelectedUserEmail(email)}
                />
              </div>
              <div className="rounded-2xl border border-slate-200/70 bg-white/50 p-3 dark:border-white/10 dark:bg-white/5">
                <div className="flex items-center justify-between">
                  <div className="text-sm font-semibold">Active users</div>
                  <div className="text-xs text-slate-600 dark:text-slate-300">{filteredLocations.length}</div>
                </div>
                <div className="mt-2 max-h-80 space-y-2 overflow-auto pr-1">
                  {filteredLocations.length === 0 ? (
                    <div className="rounded-xl border border-slate-200/70 bg-white/40 p-3 text-xs text-slate-600 dark:border-white/10 dark:bg-white/5 dark:text-slate-300">
                      No active users match your search.
                    </div>
                  ) : (
                    filteredLocations.map((l) => {
                      const active = String(l.user_email) === String(selectedUserEmail)
                      return (
                        <button
                          key={l.user_email}
                          type="button"
                          onClick={() => setSelectedUserEmail(l.user_email)}
                          className={
                            'w-full rounded-xl border p-3 text-left text-sm transition ' +
                            (active
                              ? 'border-indigo-300 bg-indigo-500/10 text-slate-900 dark:border-indigo-500/40 dark:bg-indigo-500/10 dark:text-slate-100'
                              : 'border-slate-200/70 bg-white/40 text-slate-800 hover:bg-white/70 dark:border-white/10 dark:bg-white/5 dark:text-slate-100 dark:hover:bg-white/10')
                          }
                        >
                          <div className="truncate font-semibold">{l.user_email}</div>
                          <div className="mt-1 text-xs text-slate-600 dark:text-slate-300">{l.timestamp ? new Date(l.timestamp).toLocaleString() : '—'}</div>
                        </button>
                      )
                    })
                  )}
                </div>

                <div className="mt-3 rounded-xl border border-slate-200/70 bg-white/40 p-3 text-xs text-slate-600 dark:border-white/10 dark:bg-white/5 dark:text-slate-300">
                  Tip: click a marker (or a user) to focus.
                </div>
              </div>
            </div>
          </div>

          <div className="mt-5 grid gap-3 grid-cols-2 sm:grid-cols-4">
            <div className="rounded-2xl border border-slate-200/70 bg-white/60 p-4 dark:border-white/10 dark:bg-white/5">
              <div className="text-xs text-slate-600 dark:text-slate-300">Total</div>
              <div className="mt-1 text-xl font-bold">{stats.total}</div>
            </div>
            <div className="rounded-2xl border border-slate-200/70 bg-white/60 p-4 dark:border-white/10 dark:bg-white/5">
              <div className="text-xs text-slate-600 dark:text-slate-300">New</div>
              <div className="mt-1 text-xl font-bold">{stats.unacked}</div>
            </div>
            <div className="rounded-2xl border border-slate-200/70 bg-white/60 p-4 dark:border-white/10 dark:bg-white/5">
              <div className="text-xs text-slate-600 dark:text-slate-300">MEDIUM</div>
              <div className="mt-1 text-xl font-bold">{stats.med}</div>
            </div>
            <div className="rounded-2xl border border-slate-200/70 bg-white/60 p-4 dark:border-white/10 dark:bg-white/5">
              <div className="text-xs text-slate-600 dark:text-slate-300">HIGH</div>
              <div className="mt-1 text-xl font-bold">{stats.high}</div>
            </div>
          </div>
          {error ? (
            <div className="mt-4 rounded-xl border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-700 dark:text-red-300">{error}</div>
          ) : null}
          <div className="mt-5">
            {alerts.length === 0 ? (
              <div className="text-sm text-slate-600 dark:text-slate-300">No alerts yet.</div>
            ) : (
              <div className="space-y-3">
                {alerts.map((a) => (
                  <div key={a.id} className="rounded-2xl border border-slate-200/70 bg-white/60 p-4 dark:border-white/10 dark:bg-white/5">
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                      <div className="min-w-0">
                        <div className="flex flex-wrap items-center gap-2">
                          {riskPill(a.risk_level)}
                          <div className="text-sm font-semibold text-slate-900 dark:text-slate-100">{a.user_email}</div>
                          <div className="text-sm text-slate-600 dark:text-slate-300">• {a.location}</div>
                          <div className="text-xs text-slate-500 dark:text-slate-400">{new Date(a.created_at).toLocaleString()}</div>
                        </div>
                        <div className="mt-2 text-sm text-slate-700 dark:text-slate-200">
                          <span className="font-semibold">Primary cause:</span> {a.primary_cause || a.cause || '—'}
                        </div>
                        {a.explanation ? (
                          <details className="mt-2 rounded-xl border border-slate-200/70 bg-white/40 p-3 text-xs text-slate-700 dark:border-white/10 dark:bg-white/5 dark:text-slate-200">
                            <summary className="cursor-pointer select-none font-semibold">Explanation</summary>
                            <pre className="mt-2 whitespace-pre-wrap font-mono leading-5">{a.explanation}</pre>
                          </details>
                        ) : null}
                        <div className="mt-2 text-xs text-slate-600 dark:text-slate-300">Event time: <span className="font-mono">{fmtTime(a.event_time_seconds || 0)}</span>
                          <span className="text-slate-400"> • </span> Score: <span className="font-mono">{fmtScore(a.risk_score)}</span>
                          <span className="text-slate-400"> • </span> Confidence: <span className="font-mono">{fmtConfidence(a.confidence)}</span>
                          <span className="text-slate-400"> • </span>Status:{' '}
                          <span className={a.acknowledged_at ? 'font-semibold text-emerald-700 dark:text-emerald-300' : 'font-semibold text-amber-700 dark:text-amber-300'}>{a.acknowledged_at ? 'ACK' : 'NEW'}</span>
                        </div>
                      </div>
                      <div className="flex shrink-0 flex-col gap-2">
                        <VoiceControls alert={a} />
                        <button type="button" disabled={Boolean(a.acknowledged_at) || busyId === a.id} onClick={() => onAck(a.id)} className="rounded-xl bg-indigo-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-700 disabled:opacity-60">{Boolean(a.acknowledged_at) ? 'Acknowledged' : busyId === a.id ? 'Ack…' : 'Acknowledge'}</button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
