import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { DEFAULTS } from '../auth/config.js';
import { analyzeVideo, stopLocation, updateLocation } from '../auth/api.js';
import { clearSession, getSession } from '../auth/session.js';
import ThemeToggle from '../components/ThemeToggle.jsx';

export default function UserDashboard() {
  const nav = useNavigate();
  const session = getSession();
  const [location, setLocation] = useState(DEFAULTS.defaultLocation);
  const [locationTouched, setLocationTouched] = useState(false);
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [phone, setPhone] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [sampleEverySeconds, setSampleEverySeconds] = useState(0.2);
  const [thresholdLow, setThresholdLow] = useState(0.0008);
  const [thresholdMedium, setThresholdMedium] = useState(0.0012);
  const [thresholdHigh, setThresholdHigh] = useState(0.0016);
  const [showAllSamples, setShowAllSamples] = useState(true);
  const userEmail = session?.email || '';
  const samples = Array.isArray(result?.samples) ? result.samples : [];
  const filteredSamples = useMemo(() => {
    if (showAllSamples) return samples;
    return samples.filter(s => String(s.riskLevel) !== 'NONE');
  }, [samples, showAllSamples]);

  const [isSharing, setIsSharing] = useState(false);
  const [locationStatus, setLocationStatus] = useState('');
  const lastCoordsRef = useRef(null);

  const mapsApiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;

  function getPositionOnce() {
    return new Promise((resolve, reject) => {
      if (!navigator.geolocation) {
        reject(new Error('Geolocation not supported'));
        return;
      }
      navigator.geolocation.getCurrentPosition(
        pos => resolve(pos),
        err => reject(err),
        { enableHighAccuracy: true, maximumAge: 5000, timeout: 15000 }
      );
    });
  }

  async function reverseGeocode(lat, lng) {
    if (!mapsApiKey) return null;
    const url = new URL('https://maps.googleapis.com/maps/api/geocode/json');
    url.searchParams.set('latlng', `${lat},${lng}`);
    url.searchParams.set('key', mapsApiKey);
    const res = await fetch(url.toString());
    if (!res.ok) return null;
    const data = await res.json();
    const first = Array.isArray(data?.results) ? data.results[0] : null;
    return first?.formatted_address || null;
  }

  useEffect(() => {
    if (!userEmail) return;
    if (!navigator.geolocation) {
      setLocationStatus('Geolocation not supported in this browser');
      return;
    }

    let cancelled = false;
    const tryAutoStart = async () => {
      try {
        if (!navigator.permissions?.query) return;
        const perm = await navigator.permissions.query({ name: 'geolocation' });
        if (cancelled) return;
        if (perm.state === 'granted') setIsSharing(true);
      } catch {
        alert('Error checking geolocation permissions');
      }
    };
    tryAutoStart();

    return () => {
      cancelled = true;
    };
  }, [userEmail]);

  useEffect(() => {
    if (!userEmail) {
      setLocationStatus('');
      return;
    }
    if (!navigator.geolocation) {
      setLocationStatus(isSharing ? 'Geolocation not supported' : '');
      return;
    }

    let watchId = null;

    if (isSharing) {
      setLocationStatus('Requesting GPS permission…');
      watchId = navigator.geolocation.watchPosition(
        pos => {
          const lat = pos.coords.latitude;
          const lng = pos.coords.longitude;
          lastCoordsRef.current = { lat, lng };
          updateLocation(userEmail, lat, lng)
            .then(() => setLocationStatus('Last sent: ' + new Date().toLocaleTimeString()))
            .catch(e => setLocationStatus('Error: ' + e.message));
        },
        err => {
          setLocationStatus('GPS Error: ' + err.message);
        },
        { enableHighAccuracy: true, maximumAge: 5000, timeout: 15000 }
      );
    } else {
      stopLocation(userEmail).catch(() => {});
      setLocationStatus('');
    }

    return () => {
      if (watchId !== null) navigator.geolocation.clearWatch(watchId);
    };
  }, [isSharing, userEmail]);

  const stats = useMemo(() => {
    const counts = { NONE: 0, LOW: 0, MEDIUM: 0, HIGH: 0 };
    for (const s of samples) {
      const k = String(s.riskLevel);
      if (counts[k] === undefined) continue;
      counts[k] += 1;
    }
    return {
      total: samples.length,
      ...counts,
    };
  }, [samples]);

  function riskPill(level) {
    const v = String(level || 'NONE');
    const base = 'inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold ring-1';
    if (v === 'HIGH')
      return (
        <span className={`${base} bg-red-500/10 text-red-700 ring-red-500/20 dark:text-red-300`}>
          HIGH
        </span>
      );
    if (v === 'MEDIUM')
      return (
        <span
          className={`${base} bg-amber-500/10 text-amber-700 ring-amber-500/20 dark:text-amber-300`}
        >
          MEDIUM
        </span>
      );
    if (v === 'LOW')
      return (
        <span
          className={`${base} bg-emerald-500/10 text-emerald-700 ring-emerald-500/20 dark:text-emerald-300`}
        >
          LOW
        </span>
      );
    return (
      <span
        className={`${base} bg-slate-500/10 text-slate-700 ring-slate-500/20 dark:text-slate-300`}
      >
        NONE
      </span>
    );
  }

  function fmtTime(sec) {
    const s = Math.max(0, Number(sec || 0));
    const hh = String(Math.floor(s / 3600)).padStart(2, '0');
    const mm = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
    const ss = String((s % 60).toFixed(1)).padStart(4, '0');
    return `${hh}:${mm}:${ss}`;
  }

  async function onAnalyze(e) {
    e.preventDefault();
    setError('');
    setResult(null);

    if (!file) {
      setError('Please choose a video file to upload.');
      return;
    }

    if (!phone || !/^\d{10}$/.test(phone)) {
      setError('Please enter a valid 10-digit phone number.');
      return;
    }

    let loc = (location || '').trim() || DEFAULTS.defaultLocation;

    // If GPS is available and the user grants access, override the alert location
    // with a reverse-geocoded place name (so it's not the default text).
    try {
      const pos = await getPositionOnce();
      const lat = pos.coords.latitude;
      const lng = pos.coords.longitude;
      lastCoordsRef.current = { lat, lng };
      await updateLocation(userEmail, lat, lng);
      const name = await reverseGeocode(lat, lng);
      if (name) {
        loc = name;
        if (!locationTouched) setLocation(name);
      }
    } catch {
      // Permission denied / GPS unavailable: keep the current location string.
    }

    setBusy(true);
    try {
      const data = await analyzeVideo({
        file,
        userEmail,
        phone,
        location: loc,
        options: {
          sampleEverySeconds: Number(sampleEverySeconds),
          thresholdLow: Number(thresholdLow),
          thresholdMedium: Number(thresholdMedium),
          thresholdHigh: Number(thresholdHigh),
          includeLosses: true,
        },
      });
      setResult(data);
    } catch (err) {
      setError(err?.message || String(err));
    } finally {
      setBusy(false);
    }
  }

  function logout() {
    clearSession();
    nav('/');
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-sky-100 via-slate-50 to-white text-slate-900 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950 dark:text-slate-100">
      <div className="sticky top-0 z-30 border-b border-slate-200/70 bg-white/70 backdrop-blur dark:border-white/10 dark:bg-slate-950/40">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
          <div>
            <div className="text-sm font-semibold">User Dashboard</div>
            <div className="text-xs text-slate-600 dark:text-slate-300">
              Logged in as: {userEmail}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <ThemeToggle />
            <button
              onClick={logout}
              type="button"
              className="rounded-xl border border-slate-200 bg-white/70 px-3 py-2 text-sm font-semibold text-slate-800 hover:bg-white dark:border-white/10 dark:bg-white/10 dark:text-slate-100 dark:hover:bg-white/15"
            >
              Logout
            </button>
          </div>
        </div>
      </div>
      <div className="mx-auto max-w-6xl px-4 py-6">
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="space-y-6">
            <div className="rounded-2xl border border-slate-200/70 bg-white/70 p-6 shadow-sm backdrop-blur dark:border-white/10 dark:bg-white/5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h2 className="text-lg font-bold">
                    AI-Based Public Safety Monitoring & Risk Detection
                  </h2>
                  <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
                    Upload a video clip to detect abnormal crowd-level motion patterns and generate
                    a timestamped risk timeline.
                  </p>
                </div>
                <div className="text-xs text-slate-600 dark:text-slate-300">
                  Default location:{' '}
                  <span className="font-semibold">{DEFAULTS.defaultLocation}</span>
                </div>
              </div>
              <div className="mt-4 grid gap-3 sm:grid-cols-3">
                <div className="rounded-2xl border border-slate-200/70 bg-white/60 p-4 dark:border-white/10 dark:bg-white/5">
                  <div className="text-xs text-slate-600 dark:text-slate-300">Detection</div>
                  <div className="mt-1 text-sm font-semibold">Abnormal motion patterns</div>
                </div>
                <div className="rounded-2xl border border-slate-200/70 bg-white/60 p-4 dark:border-white/10 dark:bg-white/5">
                  <div className="text-xs text-slate-600 dark:text-slate-300">Output</div>
                  <div className="mt-1 text-sm font-semibold">Risk timeline + summary</div>
                </div>
                <div className="rounded-2xl border border-slate-200/70 bg-white/60 p-4 dark:border-white/10 dark:bg-white/5">
                  <div className="text-xs text-slate-600 dark:text-slate-300">Alerting</div>
                  <div className="mt-1 text-sm font-semibold">Police notified on MEDIUM/HIGH</div>
                </div>
              </div>
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <div className="rounded-2xl border border-slate-200/70 bg-white/60 p-4 text-sm text-slate-700 dark:border-white/10 dark:bg-white/5 dark:text-slate-200">
                  <div className="text-xs font-semibold text-slate-600 dark:text-slate-300">
                    Background
                  </div>
                  <div className="mt-2">
                    Crowded environments can escalate quickly. Automated monitoring supports faster
                    detection and response.
                  </div>
                </div>
                <div className="rounded-2xl border border-slate-200/70 bg-white/60 p-4 text-sm text-slate-700 dark:border-white/10 dark:bg-white/5 dark:text-slate-200">
                  <div className="text-xs font-semibold text-slate-600 dark:text-slate-300">
                    Scope
                  </div>
                  <div className="mt-2">
                    Focuses on crowd behavior and risk classification. No face recognition or
                    individual identification.
                  </div>
                </div>
              </div>
              <div className="mt-6 rounded-2xl border border-blue-200/70 bg-blue-50/50 p-4 dark:border-blue-900/30 dark:bg-blue-900/10">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-sm font-bold text-slate-800 dark:text-slate-200">
                      Emergency Location Sharing
                    </div>
                    <div className="text-xs text-slate-600 dark:text-slate-400">
                      Share your live location with police.
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => setIsSharing(!isSharing)}
                    className={`rounded-xl px-4 py-2 text-sm font-semibold shadow-sm transition-colors ${isSharing ? 'bg-red-100 text-red-700 hover:bg-red-200 border border-red-200 dark:bg-red-900/30 dark:text-red-300 dark:border-red-800' : 'bg-blue-600 text-white hover:bg-blue-700'}`}
                  >
                    {isSharing ? 'Stop Sharing' : 'Share Location'}
                  </button>
                </div>
                {isSharing && (
                  <div className="mt-2 flex items-center gap-2 text-xs text-blue-700 dark:text-blue-300">
                    <span className="relative flex h-2 w-2">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500"></span>
                    </span>
                    {locationStatus || 'Initializing GPS...'}
                  </div>
                )}
              </div>
              <form onSubmit={onAnalyze} className="mt-5 space-y-4">
                <div>
                  <label className="text-sm font-semibold">Phone Number</label>
                  <input
                    className="mt-2 w-full rounded-xl border border-slate-200 bg-white/80 px-3 py-2 outline-none ring-indigo-500 focus:ring-2 dark:border-white/10 dark:bg-slate-950/40"
                    value={phone}
                    onChange={e => setPhone(e.target.value.replace(/[^\d]/g, ''))}
                    placeholder="Enter 10-digit phone number"
                    maxLength={10}
                  />
                </div>
                <div>
                  <label className="text-sm font-semibold">Location</label>
                  <input
                    className="mt-2 w-full rounded-xl border border-slate-200 bg-white/80 px-3 py-2 outline-none ring-indigo-500 focus:ring-2 dark:border-white/10 dark:bg-slate-950/40"
                    value={location}
                    onChange={e => {
                      setLocationTouched(true);
                      setLocation(e.target.value);
                    }}
                    placeholder={DEFAULTS.defaultLocation}
                  />
                </div>
                <div>
                  <label className="text-sm font-semibold">Video file</label>
                  <input
                    className="mt-2 w-full rounded-xl border border-slate-200 bg-white/80 px-3 py-2 outline-none dark:border-white/10 dark:bg-slate-950/40"
                    type="file"
                    accept="video/mp4,video/avi,video/mov,video/mkv"
                    onChange={e => setFile(e.target.files?.[0] || null)}
                  />
                </div>
                <details className="rounded-xl border border-slate-200/70 bg-white/60 p-4 dark:border-white/10 dark:bg-white/5">
                  <summary className="cursor-pointer text-sm font-semibold">
                    Advanced settings (model thresholds & sampling)
                  </summary>
                  <div className="mt-4 grid gap-4 sm:grid-cols-2">
                    <div>
                      <label className="text-sm font-semibold">Sample every (seconds)</label>
                      <input
                        className="mt-2 w-full rounded-xl border border-slate-200 bg-white/80 px-3 py-2 outline-none ring-indigo-500 focus:ring-2 dark:border-white/10 dark:bg-slate-950/40"
                        value={sampleEverySeconds}
                        onChange={e => setSampleEverySeconds(e.target.value)}
                      />
                      <div className="mt-1 text-xs text-slate-600 dark:text-slate-300">
                        Recommended: <span className="font-mono">0.2</span> (~5 samples/sec).
                      </div>
                    </div>
                    <div>
                      <label className="text-sm font-semibold">Thresholds (LOW / MED / HIGH)</label>
                      <div className="mt-2 grid grid-cols-3 gap-2">
                        <input
                          className="w-full rounded-xl border border-slate-200 bg-white/80 px-3 py-2 outline-none ring-indigo-500 focus:ring-2 dark:border-white/10 dark:bg-slate-950/40"
                          value={thresholdLow}
                          onChange={e => setThresholdLow(e.target.value)}
                        />
                        <input
                          className="w-full rounded-xl border border-slate-200 bg-white/80 px-3 py-2 outline-none ring-indigo-500 focus:ring-2 dark:border-white/10 dark:bg-slate-950/40"
                          value={thresholdMedium}
                          onChange={e => setThresholdMedium(e.target.value)}
                        />
                        <input
                          className="w-full rounded-xl border border-slate-200 bg-white/80 px-3 py-2 outline-none ring-indigo-500 focus:ring-2 dark:border-white/10 dark:bg-slate-950/40"
                          value={thresholdHigh}
                          onChange={e => setThresholdHigh(e.target.value)}
                        />
                      </div>
                    </div>
                  </div>
                </details>
                <button
                  className="w-full rounded-xl bg-gradient-to-r from-indigo-600 to-emerald-500 px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:opacity-95 disabled:opacity-60"
                  type="submit"
                  disabled={busy}
                >
                  {busy ? 'Analyzing…' : 'Analyze Risk'}
                </button>
              </form>
              {error ? (
                <div className="mt-4 rounded-xl border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-700 dark:text-red-300">
                  {error}
                </div>
              ) : null}
            </div>
          </div>
          <div className="space-y-6">
            <div className="rounded-2xl border border-slate-200/70 bg-white/70 p-6 shadow-sm backdrop-blur dark:border-white/10 dark:bg-white/5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h2 className="text-lg font-bold">Risk report</h2>
                  <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
                    Summary metrics and the first detected anomaly timestamp.
                  </p>
                </div>
                {result?.riskLevel ? riskPill(result.riskLevel) : null}
              </div>
              {!result ? (
                <div className="mt-6 text-sm text-slate-600 dark:text-slate-300">
                  Upload a video to generate the risk report.
                </div>
              ) : (
                <>
                  <div className="mt-5 grid gap-3 sm:grid-cols-4">
                    <div className="rounded-2xl border border-slate-200/70 bg-white/60 p-4 dark:border-white/10 dark:bg-white/5">
                      <div className="text-xs text-slate-600 dark:text-slate-300">Windows</div>
                      <div className="mt-1 text-xl font-bold">{stats.total}</div>
                    </div>
                    <div className="rounded-2xl border border-slate-200/70 bg-white/60 p-4 dark:border-white/10 dark:bg-white/5">
                      <div className="text-xs text-slate-600 dark:text-slate-300">LOW</div>
                      <div className="mt-1 text-xl font-bold">{stats.LOW}</div>
                    </div>
                    <div className="rounded-2xl border border-slate-200/70 bg-white/60 p-4 dark:border-white/10 dark:bg-white/5">
                      <div className="text-xs text-slate-600 dark:text-slate-300">MEDIUM</div>
                      <div className="mt-1 text-xl font-bold">{stats.MEDIUM}</div>
                    </div>
                    <div className="rounded-2xl border border-slate-200/70 bg-white/60 p-4 dark:border-white/10 dark:bg-white/5">
                      <div className="text-xs text-slate-600 dark:text-slate-300">HIGH</div>
                      <div className="mt-1 text-xl font-bold">{stats.HIGH}</div>
                    </div>
                  </div>
                  <div className="mt-5 grid gap-3 sm:grid-cols-2">
                    <div className="rounded-2xl border border-slate-200/70 bg-white/60 p-4 dark:border-white/10 dark:bg-white/5">
                      <div className="text-xs text-slate-600 dark:text-slate-300">Event time</div>
                      <div className="mt-1 font-mono text-sm">
                        {fmtTime(result.eventTimeSeconds || 0)}
                      </div>
                    </div>
                    <div className="rounded-2xl border border-slate-200/70 bg-white/60 p-4 dark:border-white/10 dark:bg-white/5">
                      <div className="text-xs text-slate-600 dark:text-slate-300">
                        Max loss / Mean loss
                      </div>
                      <div className="mt-1 text-sm">
                        <span className="font-mono">
                          {Number(result.maxLoss ?? result.riskScore).toFixed(6)}
                        </span>
                        <span className="text-slate-500 dark:text-slate-400"> • </span>
                        <span className="font-mono">{Number(result.meanLoss ?? 0).toFixed(6)}</span>
                      </div>
                    </div>
                  </div>
                  {result.alertCreated ? (
                    <div className="mt-4 rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-3 text-sm text-emerald-800 dark:text-emerald-200">
                      Police alerted for <span className="font-semibold">{result.userEmail}</span>{' '}
                      at <span className="font-semibold">{result.location}</span>.
                    </div>
                  ) : (
                    <div className="mt-4 text-sm text-slate-600 dark:text-slate-300">
                      No police alert sent (risk stayed below MEDIUM).
                    </div>
                  )}
                </>
              )}
            </div>
            <div className="rounded-2xl border border-slate-200/70 bg-white/70 p-6 shadow-sm backdrop-blur dark:border-white/10 dark:bg-white/5">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <h3 className="text-base font-bold">Timeline</h3>
                  <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
                    Risk level and cause for each analysis window.
                  </p>
                </div>
                <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-200">
                  <input
                    type="checkbox"
                    checked={showAllSamples}
                    onChange={e => setShowAllSamples(e.target.checked)}
                  />
                  Show NONE
                </label>
              </div>
              {!result ? (
                <div className="mt-4 text-sm text-slate-600 dark:text-slate-300">
                  No results yet.
                </div>
              ) : (
                <div className="mt-4 max-h-[460px] space-y-3 overflow-auto pr-2">
                  {filteredSamples.length === 0 ? (
                    <div className="text-sm text-slate-600 dark:text-slate-300">
                      No entries for the current filter.
                    </div>
                  ) : (
                    filteredSamples.map((s, idx) => (
                      <div
                        key={idx}
                        className="rounded-2xl border border-slate-200/70 bg-white/60 p-4 dark:border-white/10 dark:bg-white/5"
                      >
                        <div className="flex items-center justify-between gap-3">
                          {riskPill(s.riskLevel)}
                          <div className="font-mono text-xs text-slate-600 dark:text-slate-300">
                            {fmtTime(s.timeSeconds)}
                          </div>
                        </div>
                        <div className="mt-2 text-sm">
                          <span className="font-semibold">Cause:</span> {s.cause}
                        </div>
                        <div className="mt-2 text-xs text-slate-600 dark:text-slate-300">
                          loss=<span className="font-mono">{Number(s.loss ?? 0).toFixed(6)}</span>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
