import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { DEFAULTS } from '../auth/config.js'
import { setSession } from '../auth/session.js'
import ThemeToggle from '../components/ThemeToggle.jsx'

export default function Login() {
  const nav = useNavigate()
  const [tab, setTab] = useState('user')
  const [formError, setFormError] = useState('')
  const [userEmail, setUserEmail] = useState(DEFAULTS.user.email)
  const [userPassword, setUserPassword] = useState(DEFAULTS.user.password)
  const [policeEmail, setPoliceEmail] = useState(DEFAULTS.police.email)
  const [policeId, setPoliceId] = useState(DEFAULTS.police.policeId)
  const [policePassword, setPolicePassword] = useState(DEFAULTS.police.password)
  const help = useMemo(() => {
    if (tab === 'police') {
      return 'Police login requires Email + Police ID + Password.'
    }
    return 'User login requires Email + Password.'
  }, [tab])

  function onSubmit(e) {
    e.preventDefault()
    setFormError('')

    if (tab === 'user') {
      if (userEmail === DEFAULTS.user.email && userPassword === DEFAULTS.user.password) {
        setSession({ role: 'user', email: userEmail })
        nav('/user')
        return
      }
      setFormError('Invalid user credentials.')
      return
    }

    if (
      policeEmail === DEFAULTS.police.email &&
      policeId === DEFAULTS.police.policeId &&
      policePassword === DEFAULTS.police.password
    ) {
      setSession({ role: 'police', email: policeEmail, policeId })
      nav('/police')
      return
    }
    setFormError('Invalid police credentials.')
  }

  return (
    <div className="min-h-screen bg-[radial-gradient(900px_600px_at_10%_10%,rgba(99,102,241,0.18),transparent_60%),radial-gradient(800px_500px_at_90%_30%,rgba(16,185,129,0.14),transparent_55%)] bg-slate-50 text-slate-900 dark:bg-[radial-gradient(900px_600px_at_10%_10%,rgba(99,102,241,0.16),transparent_60%),radial-gradient(800px_500px_at_90%_30%,rgba(16,185,129,0.12),transparent_55%)] dark:bg-slate-950 dark:text-slate-100">
      <div className="mx-auto flex min-h-screen max-w-6xl items-center justify-center px-3 sm:px-4 py-6 sm:py-10">
        <div className="w-full max-w-5xl">
          <div className="mb-4 sm:mb-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
            <div className="w-full sm:w-auto">
              <div className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white/70 px-3 py-1 text-xs font-semibold text-slate-700 shadow-sm backdrop-blur dark:border-white/10 dark:bg-white/5 dark:text-slate-200">
                <span className="h-2 w-2 rounded-full bg-emerald-500" />Logic-centric • Algorithm-focused
              </div>
              <h1 className="mt-3 text-2xl sm:text-3xl lg:text-4xl font-extrabold tracking-tight">AI-Based Public Safety Monitoring & Risk Detection</h1>
              <p className="mt-2 max-w-2xl text-xs sm:text-sm text-slate-600 dark:text-slate-300">Analyze video input to detect abnormal crowd-level behavior and classify public safety risk in near real time.</p>
            </div>
            <ThemeToggle />
          </div>
          <div className="grid gap-4 sm:gap-6 lg:grid-cols-2">
            <div className="space-y-4 sm:space-y-6">
              <div className="rounded-3xl border border-slate-200/70 bg-white/70 p-4 sm:p-6 shadow-sm backdrop-blur dark:border-white/10 dark:bg-white/5">
                <div className="text-sm font-semibold">Background</div>
                <p className="mt-2 text-xs sm:text-sm text-slate-600 dark:text-slate-300">
                  In crowded public environments (markets, festivals, metro stations), incidents like panic, stampedes, sudden crowd movement, or accidents can escalate quickly if not detected early.
                </p>
                <div className="mt-4 rounded-2xl border border-slate-200/70 bg-white/60 p-3 sm:p-4 text-xs sm:text-sm text-slate-700 dark:border-white/10 dark:bg-white/5 dark:text-slate-200">
                  <div className="text-xs font-semibold text-slate-600 dark:text-slate-300">Traditional CCTV monitoring is:</div>
                  <ul className="mt-2 list-disc space-y-1 pl-5 text-xs sm:text-sm text-slate-700 dark:text-slate-200">
                    <li>Slow to respond</li>
                    <li>Prone to human error</li>
                    <li>Difficult to scale across multiple camera feeds</li>
                  </ul>
                </div>

                <div className="mt-4 text-sm font-semibold">System capabilities</div>
                <div className="mt-4 grid gap-3 grid-cols-1 sm:grid-cols-2">
                  <div className="rounded-2xl border border-slate-200/70 bg-white/60 p-4 dark:border-white/10 dark:bg-white/5">
                    <div className="text-xs text-slate-600 dark:text-slate-300">Detection</div>
                    <div className="mt-1 text-sm font-semibold">Abnormal crowd motion</div>
                    <div className="mt-1 text-xs text-slate-600 dark:text-slate-300">Learns normal patterns and flags deviations.</div>
                  </div>
                  <div className="rounded-2xl border border-slate-200/70 bg-white/60 p-4 dark:border-white/10 dark:bg-white/5">
                    <div className="text-xs text-slate-600 dark:text-slate-300">Timeline</div>
                    <div className="mt-1 text-sm font-semibold">Time-stamped risk</div>
                    <div className="mt-1 text-xs text-slate-600 dark:text-slate-300">Shows risk level with cause for each window.</div>
                  </div>
                  <div className="rounded-2xl border border-slate-200/70 bg-white/60 p-4 dark:border-white/10 dark:bg-white/5">
                    <div className="text-xs text-slate-600 dark:text-slate-300">Alerting</div>
                    <div className="mt-1 text-sm font-semibold">MEDIUM/HIGH triggers</div>
                    <div className="mt-1 text-xs text-slate-600 dark:text-slate-300">Police dashboard receives persistent alerts.</div>
                  </div>
                  <div className="rounded-2xl border border-slate-200/70 bg-white/60 p-4 dark:border-white/10 dark:bg-white/5">
                    <div className="text-xs text-slate-600 dark:text-slate-300">Audit</div>
                    <div className="mt-1 text-sm font-semibold">Acknowledgements</div>
                    <div className="mt-1 text-xs text-slate-600 dark:text-slate-300">Track status: NEW → ACK.</div>
                  </div>
                </div>
              </div>
              <div className="rounded-3xl border border-slate-200/70 bg-white/70 p-4 sm:p-6 shadow-sm backdrop-blur dark:border-white/10 dark:bg-white/5">
                <div className="flex flex-col sm:flex-row items-start justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold">Role selection</div>
                    <p className="mt-1 text-xs sm:text-sm text-slate-600 dark:text-slate-300">Choose your workspace. Access is role-based.</p>
                  </div>
                  <div className="text-xs text-slate-600 dark:text-slate-300">User / Police</div>
                </div>
                <div className="mt-4 grid grid-cols-2 rounded-2xl border border-slate-200 bg-slate-50 p-1 dark:border-white/10 dark:bg-white/5">
                  <button type="button" onClick={() => setTab('user')} className={tab === 'user' ? 'rounded-xl bg-white px-3 py-2 text-xs sm:text-sm font-semibold shadow-sm dark:bg-white/10' : 'rounded-xl px-3 py-2 text-xs sm:text-sm font-semibold text-slate-600 hover:text-slate-900 dark:text-slate-300 dark:hover:text-white'}>User</button>
                  <button type="button" onClick={() => setTab('police')} className={tab === 'police' ? 'rounded-xl bg-white px-3 py-2 text-xs sm:text-sm font-semibold shadow-sm dark:bg-white/10' : 'rounded-xl px-3 py-2 text-xs sm:text-sm font-semibold text-slate-600 hover:text-slate-900 dark:text-slate-300 dark:hover:text-white'}> Police</button>
                </div>
                <div className="mt-4 grid gap-3 grid-cols-1 sm:grid-cols-2">
                  <div className="rounded-2xl border border-slate-200/70 bg-white/60 p-3 sm:p-4 dark:border-white/10 dark:bg-white/5">
                    <div className="text-xs text-slate-600 dark:text-slate-300">User workspace</div>
                    <div className="mt-1 text-sm font-semibold">Upload & Risk Report</div>
                    <div className="mt-1 text-xs text-slate-600 dark:text-slate-300">Submit video and review the timeline.</div>
                  </div>
                  <div className="rounded-2xl border border-slate-200/70 bg-white/60 p-3 sm:p-4 dark:border-white/10 dark:bg-white/5">
                    <div className="text-xs text-slate-600 dark:text-slate-300">Police workspace</div>
                    <div className="mt-1 text-sm font-semibold">Alerts & Acknowledgement</div>
                    <div className="mt-1 text-xs text-slate-600 dark:text-slate-300">Monitor incidents and acknowledge responses.</div>
                  </div>
                </div>
                <div className="mt-4 text-xs sm:text-sm text-slate-600 dark:text-slate-300">{help}</div>
              </div>
            </div>
            <div className="flex flex-col rounded-3xl border border-slate-200/70 bg-white/70 p-4 sm:p-6 shadow-sm backdrop-blur dark:border-white/10 dark:bg-white/5">
              <div className="flex flex-col sm:flex-row items-start justify-between gap-3">
                <div>
                  <div className="text-sm font-semibold">Sign in</div>
                  <div className="mt-1 text-xs sm:text-sm text-slate-600 dark:text-slate-300">Continue as <span className="font-semibold">{tab === 'user' ? 'User (Upload & Report)' : 'Police (Alerts & Response)'}</span></div>
                </div>
                <div className="rounded-full border border-slate-200 bg-white/70 px-3 py-1 text-xs font-semibold text-slate-700 dark:border-white/10 dark:bg-white/5 dark:text-slate-200">{tab === 'user' ? 'User' : 'Police'} Mode</div>
              </div>
              <form onSubmit={onSubmit} className="mt-5 space-y-3 sm:space-y-4">
                  {tab === 'user' ? (
                    <>
                      <div>
                        <label className="text-sm font-semibold">Email</label>
                        <input className="mt-2 w-full rounded-xl border border-slate-200 bg-white/80 px-3 py-2 outline-none ring-indigo-500 focus:ring-2 dark:border-white/10 dark:bg-slate-950/40" value={userEmail} onChange={(e) => setUserEmail(e.target.value)} />
                      </div>
                      <div>
                        <label className="text-sm font-semibold">Password</label>
                        <input className="mt-2 w-full rounded-xl border border-slate-200 bg-white/80 px-3 py-2 outline-none ring-indigo-500 focus:ring-2 dark:border-white/10 dark:bg-slate-950/40" type="password" value={userPassword} onChange={(e) => setUserPassword(e.target.value)} />
                      </div>
                    </>
                  ) : (
                    <>
                      <div>
                        <label className="text-sm font-semibold">Email</label>
                        <input className="mt-2 w-full rounded-xl border border-slate-200 bg-white/80 px-3 py-2 outline-none ring-indigo-500 focus:ring-2 dark:border-white/10 dark:bg-slate-950/40" value={policeEmail} onChange={(e) => setPoliceEmail(e.target.value)} />
                      </div>
                      <div>
                        <label className="text-sm font-semibold">Police ID</label>
                        <input className="mt-2 w-full rounded-xl border border-slate-200 bg-white/80 px-3 py-2 outline-none ring-indigo-500 focus:ring-2 dark:border-white/10 dark:bg-slate-950/40" value={policeId} onChange={(e) => setPoliceId(e.target.value)}
                        />
                      </div>
                      <div>
                        <label className="text-sm font-semibold">Password</label>
                        <input className="mt-2 w-full rounded-xl border border-slate-200 bg-white/80 px-3 py-2 outline-none ring-indigo-500 focus:ring-2 dark:border-white/10 dark:bg-slate-950/40" type="password" value={policePassword} onChange={(e) => setPolicePassword(e.target.value)} />
                      </div>
                    </>
                  )}
                  <button className="w-full rounded-2xl bg-gradient-to-r from-indigo-600 to-emerald-500 px-4 py-2.5 sm:py-3 text-sm font-semibold text-white shadow-sm hover:opacity-95" type="submit">Continue</button>
                  {formError ? (
                    <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-3 text-xs sm:text-sm text-red-700 dark:text-red-300"> {formError}</div>
                  ) : null}
                </form>
              <div className="mt-5 rounded-2xl border border-slate-200/70 bg-white/60 p-4 sm:p-5 dark:border-white/10 dark:bg-white/5">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold">Project scope</div>
                    <div className="mt-1 text-xs text-slate-600 dark:text-slate-300">Focuses strictly on crowd-level behavior, not individual identification or surveillance.</div>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <span className="rounded-full bg-indigo-500/10 px-2.5 sm:px-3 py-1 text-xs font-semibold text-indigo-700 ring-1 ring-indigo-500/20 dark:text-indigo-300">Timeline</span>
                    <span className="rounded-full bg-emerald-500/10 px-2.5 sm:px-3 py-1 text-xs font-semibold text-emerald-700 ring-1 ring-emerald-500/20 dark:text-emerald-300">Alerts</span>
                    <span className="rounded-full bg-slate-500/10 px-2.5 sm:px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-500/20 dark:text-slate-300">Audit</span>
                  </div>
                </div>
                <div className="mt-4 grid gap-3 grid-cols-1 sm:grid-cols-2">
                  <div className="rounded-2xl border border-slate-200/70 bg-white/70 p-3 sm:p-4 dark:border-white/10 dark:bg-slate-950/20">
                    <div className="text-xs text-slate-600 dark:text-slate-300">How it works</div>
                    <div className="mt-1 text-sm font-semibold">Reconstruction error</div>
                    <div className="mt-1 text-xs text-slate-600 dark:text-slate-300">A pretrained autoencoder estimates normal motion; deviations increase loss and raise risk.</div>
                  </div>
                  <div className="rounded-2xl border border-slate-200/70 bg-white/70 p-3 sm:p-4 dark:border-white/10 dark:bg-slate-950/20">
                    <div className="text-xs text-slate-600 dark:text-slate-300">Outputs</div>
                    <div className="mt-1 text-sm font-semibold">Risk + timestamped windows</div>
                    <div className="mt-1 text-xs text-slate-600 dark:text-slate-300">Produces a time-based timeline (risk level, cause, loss) and an overall severity.</div>
                  </div>
                </div>
                <div className="mt-4 rounded-2xl border border-slate-200/70 bg-white/70 p-3 sm:p-4 text-xs text-slate-600 dark:border-white/10 dark:bg-slate-950/20 dark:text-slate-300">
                  <div className="font-semibold text-slate-800 dark:text-slate-100">Recommended input</div>
                  <div className="mt-1">Formats: MP4 / AVI / MOV / MKV • Sampling: ~0.2s • Best for fixed-camera crowd scenes.</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
