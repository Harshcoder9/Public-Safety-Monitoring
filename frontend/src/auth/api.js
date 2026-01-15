import { DEFAULTS } from './config.js'

export async function analyzeVideo({ file, userEmail, location, options }) {
  const form = new FormData()
  form.append('file', file)
  form.append('userEmail', userEmail)
  form.append('location', location)
  form.append('analyzer', 'autoencoder')

  const o = options || {}
  form.append('sampleEverySeconds', String(o.sampleEverySeconds ?? 0.2))
  form.append('thresholdLow', String(o.thresholdLow ?? 0.0008))
  form.append('thresholdMedium', String(o.thresholdMedium ?? 0.0012))
  form.append('thresholdHigh', String(o.thresholdHigh ?? 0.0016))
  form.append('includeLosses', String(o.includeLosses ?? true))

  const res = await fetch(`${DEFAULTS.apiBaseUrl}/api/analyze`, {
    method: 'POST',
    body: form,
  })

  if (!res.ok) {
    const txt = await res.text()
    throw new Error(txt || `Analyze failed (${res.status})`)
  }
  return res.json()
}

export async function fetchAlerts({ includeAcknowledged = true } = {}) {
  const url = new URL(`${DEFAULTS.apiBaseUrl}/api/alerts`)
  url.searchParams.set('includeAcknowledged', String(includeAcknowledged))

  const res = await fetch(url.toString())
  if (!res.ok) {
    const txt = await res.text()
    throw new Error(txt || `Fetch alerts failed (${res.status})`)
  }
  return res.json()
}

export async function acknowledgeAlert(alertId) {
  const res = await fetch(`${DEFAULTS.apiBaseUrl}/api/alerts/${alertId}/ack`, {
    method: 'POST',
  })
  if (!res.ok) {
    const txt = await res.text()
    throw new Error(txt || `Acknowledge failed (${res.status})`)
  }
  return res.json()
}

export async function updateLocation(userEmail, lat, lng) {
  const form = new FormData()
  form.append('userEmail', userEmail)
  form.append('latitude', String(lat))
  form.append('longitude', String(lng))

  const res = await fetch(`${DEFAULTS.apiBaseUrl}/api/location`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
     const txt = await res.text()
    throw new Error(txt || `Update location failed (${res.status})`)
  }
  return res.json()
}

export async function fetchLocations() {
  const res = await fetch(`${DEFAULTS.apiBaseUrl}/api/locations`)
  if (!res.ok) {
     const txt = await res.text()
    throw new Error(txt || `Fetch locations failed (${res.status})`)
  }
  return res.json()
}
