import React, { useEffect, useMemo, useRef, useState } from 'react'
import { GoogleMap, InfoWindow, Marker, useJsApiLoader } from '@react-google-maps/api'

const containerStyle = { width: '100%', height: '100%', minHeight: '400px', borderRadius: '1rem' }

const defaultCenter = { lat: 19.2183, lng: 72.8367 }

const DARK_MAP_STYLE = [
  { elementType: 'geometry', stylers: [{ color: '#0b1220' }] },
  { elementType: 'labels.text.stroke', stylers: [{ color: '#0b1220' }] },
  { elementType: 'labels.text.fill', stylers: [{ color: '#94a3b8' }] },
  { featureType: 'administrative', elementType: 'geometry', stylers: [{ color: '#334155' }] },
  { featureType: 'poi', elementType: 'labels.text.fill', stylers: [{ color: '#64748b' }] },
  { featureType: 'poi.park', elementType: 'geometry', stylers: [{ color: '#0f172a' }] },
  { featureType: 'road', elementType: 'geometry', stylers: [{ color: '#111827' }] },
  { featureType: 'road', elementType: 'geometry.stroke', stylers: [{ color: '#1f2937' }] },
  { featureType: 'road', elementType: 'labels.text.fill', stylers: [{ color: '#cbd5e1' }] },
  { featureType: 'water', elementType: 'geometry', stylers: [{ color: '#020617' }] },
  { featureType: 'water', elementType: 'labels.text.fill', stylers: [{ color: '#475569' }] },
]

function isFiniteNumber(v) {
  const n = Number(v)
  return Number.isFinite(n)
}

function parseTimestamp(ts) {
  if (!ts) return null
  const d = new Date(ts)
  return Number.isFinite(d.getTime()) ? d : null
}

function relativeTime(ts) {
  const d = parseTimestamp(ts)
  if (!d) return ''
  const diffMs = Date.now() - d.getTime()
  if (!Number.isFinite(diffMs)) return ''
  const sec = Math.max(0, Math.round(diffMs / 1000))
  if (sec < 60) return `${sec}s ago`
  const min = Math.round(sec / 60)
  if (min < 60) return `${min}m ago`
  const hr = Math.round(min / 60)
  return `${hr}h ago`
}

export default function Map({ locations = [], selectedUserEmail = '', onSelectUser } = {}) {
  const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY
  const mapRef = useRef(null)
  const [internalSelected, setInternalSelected] = useState('')
  const [isDark, setIsDark] = useState(() => document.documentElement.classList.contains('dark'))

  if (!apiKey || apiKey === '') {
    return (
      <div className="w-full h-96 bg-slate-100 flex flex-col gap-2 items-center justify-center rounded-xl text-slate-500 border border-slate-200 p-4 text-center">
        <p className="font-semibold">Google Maps API Key Missing</p>
        <p className="text-sm">Please add VITE_GOOGLE_MAPS_API_KEY to frontend/.env</p>
        <div className="mt-4 text-xs text-left w-full overflow-auto bg-slate-200 p-2 rounded">
          <strong>Locations to display:</strong>
          <pre>{JSON.stringify(locations, null, 2)}</pre>
        </div>
      </div>
    )
  }

  const { isLoaded, loadError } = useJsApiLoader({
    id: 'google-map-script',
    googleMapsApiKey: apiKey,
  })

  useEffect(() => {
    const el = document.documentElement
    const obs = new MutationObserver(() => {
      setIsDark(el.classList.contains('dark'))
    })
    obs.observe(el, { attributes: true, attributeFilter: ['class'] })
    return () => obs.disconnect()
  }, [])

  const cleanedLocations = useMemo(() => {
    const arr = Array.isArray(locations) ? locations : []
    return arr
      .filter((loc) => loc && isFiniteNumber(loc.latitude) && isFiniteNumber(loc.longitude))
      .map((loc) => ({
        user_email: String(loc.user_email || ''),
        latitude: Number(loc.latitude),
        longitude: Number(loc.longitude),
        timestamp: loc.timestamp,
      }))
      .filter((loc) => Boolean(loc.user_email))
  }, [locations])

  const center = useMemo(() => {
    return cleanedLocations.length > 0 ? { lat: cleanedLocations[0].latitude, lng: cleanedLocations[0].longitude } : defaultCenter
  }, [cleanedLocations])

  const selected = selectedUserEmail || internalSelected

  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    if (cleanedLocations.length === 0) return

    // Fit bounds to all markers for a polished “ops map” feel.
    try {
      const bounds = new window.google.maps.LatLngBounds()
      for (const loc of cleanedLocations) bounds.extend({ lat: loc.latitude, lng: loc.longitude })
      map.fitBounds(bounds)

      // If there is only one marker, fitBounds zooms too far sometimes.
      if (cleanedLocations.length === 1) map.setZoom(14)
    } catch {
      // ignore
    }
  }, [cleanedLocations])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    if (!selected) return
    const loc = cleanedLocations.find((l) => l.user_email === selected)
    if (!loc) return

    try {
      map.panTo({ lat: loc.latitude, lng: loc.longitude })
      if ((map.getZoom?.() || 0) < 14) map.setZoom(14)
    } catch {
      // ignore
    }
  }, [selected, cleanedLocations])

  if (loadError) {
    return (
      <div className="w-full h-96 bg-slate-100 flex flex-col gap-2 items-center justify-center rounded-xl text-slate-500 border border-slate-200 p-4 text-center">
        <p className="font-semibold">Failed to load Google Maps</p>
        <p className="text-sm">{String(loadError?.message || loadError)}</p>
      </div>
    )
  }

  if (!isLoaded) {
    return (
      <div className="w-full h-96 bg-slate-100 flex flex-col gap-2 items-center justify-center rounded-xl text-slate-500 border border-slate-200 p-4 text-center">
        <p className="font-semibold">Loading map…</p>
      </div>
    )
  }

  return (
    <div className="h-96 w-full overflow-hidden rounded-xl border border-slate-200/70 shadow-sm dark:border-white/10">
      <GoogleMap
        mapContainerStyle={containerStyle}
        center={center}
        zoom={13}
        onLoad={(m) => {
          mapRef.current = m
        }}
        onUnmount={() => {
          mapRef.current = null
        }}
        options={{
          styles: isDark ? DARK_MAP_STYLE : undefined,
          disableDefaultUI: true,
          zoomControl: true,
          fullscreenControl: false,
          clickableIcons: false,
        }}
      >
        {cleanedLocations.map((loc) => {
          const active = loc.user_email === selected
          const label = loc.user_email.includes('@') ? loc.user_email.split('@')[0] : loc.user_email
          return (
            <Marker
              key={loc.user_email}
              position={{ lat: loc.latitude, lng: loc.longitude }}
              onClick={() => {
                setInternalSelected(loc.user_email)
                if (typeof onSelectUser === 'function') onSelectUser(loc.user_email)
              }}
              title={`${loc.user_email}${loc.timestamp ? ` • ${loc.timestamp}` : ''}`}
              label={{
                text: label.slice(0, 12),
                className: active ? 'font-semibold' : '',
              }}
              icon={
                active
                  ? {
                      path: window.google.maps.SymbolPath.CIRCLE,
                      scale: 9,
                      fillColor: '#6366f1',
                      fillOpacity: 0.95,
                      strokeColor: '#ffffff',
                      strokeWeight: 2,
                    }
                  : {
                      path: window.google.maps.SymbolPath.CIRCLE,
                      scale: 7,
                      fillColor: '#22c55e',
                      fillOpacity: 0.9,
                      strokeColor: '#0f172a',
                      strokeWeight: 1,
                    }
              }
            />
          )
        })}

        {selected ? (
          (() => {
            const loc = cleanedLocations.find((l) => l.user_email === selected)
            if (!loc) return null
            return (
              <InfoWindow
                position={{ lat: loc.latitude, lng: loc.longitude }}
                onCloseClick={() => {
                  setInternalSelected('')
                  if (typeof onSelectUser === 'function') onSelectUser('')
                }}
              >
                <div className="min-w-[180px]">
                  <div className="text-sm font-semibold text-slate-900">{loc.user_email}</div>
                  <div className="mt-1 text-xs text-slate-600">
                    Last update: {relativeTime(loc.timestamp) || '—'}
                  </div>
                  <div className="mt-1 text-xs text-slate-600">
                    {loc.latitude.toFixed(5)}, {loc.longitude.toFixed(5)}
                  </div>
                </div>
              </InfoWindow>
            )
          })()
        ) : null}
      </GoogleMap>
    </div>
  )
}
