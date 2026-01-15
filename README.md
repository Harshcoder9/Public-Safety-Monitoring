# AI-Based Public Safety Monitoring & Risk Detection System

A full-stack project that analyzes crowd-level behavior from video input and produces a **timestamped risk timeline**. When risk escalates to **MEDIUM** or **HIGH**, the backend automatically creates a **police alert** (persisted on disk) that appears in the Police Dashboard.

This project focuses on **crowd motion / scene-level anomaly detection**. It does **not** perform face recognition or individual identification.

---

## Background

In crowded public environments (markets, festivals, metro stations), incidents like panic, stampedes, sudden crowd movement, or accidents can escalate quickly if not detected early.

Traditional CCTV monitoring is:
- Slow to respond
- Prone to human error
- Difficult to scale across multiple camera feeds

This system addresses that by converting video input into a **risk classification** with **time-based evidence**.

---

## (Project Objective)

Design and implement a logic-driven AI system that:
- Accepts video input (pre-recorded or simulated)
- Detects abnormal crowd behavior at the scene level
- Classifies public safety risk as **NONE / LOW / MEDIUM / HIGH**
- Produces a **timeline** of risk windows with **time + cause**
- Triggers a **police alert** when any window is **MEDIUM** or **HIGH**

---

## What You Get

### User Dashboard
- Upload video
- Configure sampling + thresholds
- View:
  - Overall risk and score
  - Event time (first MEDIUM/HIGH window)
  - Timeline entries: **Risk Level / Time / Cause**

### Police Dashboard
- Polls alerts every 3 seconds
- Displays user email + location + event time + risk score + cause
- Allows acknowledging alerts
- Alerts persist across restarts (stored in JSON)

---

## Architecture

- **Frontend**: Vite + React + Tailwind
- **Backend**: FastAPI (Python)
- **Analyzer (default in UI)**: TensorFlow/Keras autoencoder (pretrained)
- **Optional analyzer**: OpenCV optical-flow spike detector

Flow:
1. User uploads video in the frontend
2. Frontend sends a multipart request to the backend
3. Backend runs an analyzer and returns:
   - Overall risk
   - Timestamped samples
4. If any sample is MEDIUM/HIGH, backend creates an alert
5. Police dashboard polls and shows alerts

---

## Repository Structure

- [backend/app/main.py](backend/app/main.py): FastAPI routes (`/api/analyze`, `/api/alerts`, ack)
- [backend/app/analyzers/autoencoder.py](backend/app/analyzers/autoencoder.py): Autoencoder-based analysis
- [backend/app/analyzers/optical_flow.py](backend/app/analyzers/optical_flow.py): Optical-flow-based analysis
- [backend/app/storage.py](backend/app/storage.py): JSON persistence for alerts
- [backend/data/alerts.json](backend/data/alerts.json): Persisted police alerts
- [frontend](frontend): Vite/React UI (User + Police dashboards)
- [frontend/.env](frontend/.env): Demo credentials + API base URL + defaults
- [Crowd_Anomaly_Detection/AnomalyDetector.h5](Crowd_Anomaly_Detection/AnomalyDetector.h5): Pretrained model used by the autoencoder analyzer
- [Crowd_Anomaly_Detection/run_video_risk_alerts.py](Crowd_Anomaly_Detection/run_video_risk_alerts.py): Model utilities used by the backend analyzer

---

## Prerequisites

### Required
- Python 3.10+ (recommended for best TensorFlow compatibility)
- Node.js 18+ (or newer)

### Notes (Windows)
- TensorFlow installation can be heavy; ensure you have enough disk space.
- NumPy is pinned to `<2` because many TensorFlow builds still require it.

---

## Setup (Backend)

From the repo root:

1) Create/activate a virtual environment (optional but recommended)

PowerShell:
- Create: `python -m venv .venv`
- Activate: `.\.venv\Scripts\Activate.ps1`

2) Install dependencies

- Backend API deps: `pip install -r backend/requirements.txt`
- Model/analyzer deps: `pip install -r Crowd_Anomaly_Detection/requirements.txt`

3) Run the API

- `python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000`

Health check:
- Open `http://127.0.0.1:8000/api/health`

Uploads are stored under:
- [backend/uploads](backend/uploads)

Alerts persist to:
- [backend/data/alerts.json](backend/data/alerts.json)

---

## Setup (Frontend)

1) Go to the frontend folder
- `cd frontend`

2) Install dependencies
- `npm install`

3) Configure environment variables

Edit demo values in:
- [frontend/.env](frontend/.env)

4) Run the dev server
- `npm run dev`

Open:
- `http://localhost:5173`

---

## Environment Variables (Frontend)

Configured in [frontend/.env](frontend/.env):

- `VITE_API_BASE_URL` (default: `http://127.0.0.1:8000`)
- `VITE_DEFAULT_LOCATION` (default: `kandivali`)
- `VITE_FORCE_LOCATION` (`true` or `false`)

Demo credentials:
- `VITE_USER_EMAIL`
- `VITE_USER_PASSWORD`
- `VITE_POLICE_EMAIL`
- `VITE_POLICE_ID`
- `VITE_POLICE_PASSWORD`

---

## API Reference

### `GET /api/health`
Returns server status.

### `POST /api/analyze`
Multipart form upload.

Form fields (common):
- `file` (video: `.mp4`, `.avi`, `.mov`, `.mkv`)
- `userEmail` (string)
- `location` (string; default `kandivali`)
- `analyzer` (`autoencoder` or `optical_flow`)

Autoencoder parameters:
- `sampleEverySeconds` (default `0.2`)
- `thresholdLow` (default `0.0008`)
- `thresholdMedium` (default `0.0012`)
- `thresholdHigh` (default `0.0016`)
- `includeLosses` (`true/false`)

Optical-flow parameters:
- `processFps` (default `5.0`)
- `minConsecutive` (default `1`)
- `zLow` (default `3.0`)
- `zMed` (default `5.0`)
- `zHigh` (default `7.0`)

Response highlights:
- `riskLevel`: `NONE | LOW | MEDIUM | HIGH`
- `riskScore`: numeric score
- `eventTimeSeconds`: first time a MEDIUM/HIGH window occurred
- `samples`: array of timeline entries with `riskLevel`, `timeSeconds`, `cause` (and optional diagnostics)
- `alertCreated`: `true` if risk was MEDIUM/HIGH
- `alert`: created alert object (when `alertCreated=true`)

Example (curl):
- `curl -X POST "http://127.0.0.1:8000/api/analyze" -F "file=@your_video.mp4" -F "userEmail=user@example.com" -F "location=kandivali" -F "analyzer=autoencoder" -F "sampleEverySeconds=0.2"`

### `GET /api/alerts?includeAcknowledged=true|false`
Returns persisted alerts.

### `POST /api/alerts/{id}/ack`
Marks an alert as acknowledged.

---

## Risk + Alerts Rules

- The analyzer returns a timeline of sampled windows.
- If **any** sample window is **MEDIUM** or **HIGH**, the backend creates a police alert.
- The alert includes:
  - user email
  - location
  - risk level
  - risk score
  - event time seconds
  - cause

---

## Troubleshooting

### Frontend loads but API calls fail
- Verify backend is running at `http://127.0.0.1:8000`
- Verify `VITE_API_BASE_URL` in [frontend/.env](frontend/.env)
- If you change `.env`, restart `npm run dev`

### CORS error in browser console
- Default backend CORS allows `http://localhost:5173` and `http://127.0.0.1:5173`.
- If your frontend runs on a different origin, update CORS in [backend/app/main.py](backend/app/main.py).

### TensorFlow import errors
- Ensure you installed [Crowd_Anomaly_Detection/requirements.txt](Crowd_Anomaly_Detection/requirements.txt)
- Ensure NumPy remains `<2`

### Alerts not persisting
- Confirm [backend/data](backend/data) exists and is writable
- Alerts are stored in [backend/data/alerts.json](backend/data/alerts.json)

---

## Security & Privacy Notes

- This is a demo system with simple credential checks in the frontend.
- Do not deploy as-is on the public internet.
- The intended scope is **crowd-level risk detection**, not personal surveillance.

---

