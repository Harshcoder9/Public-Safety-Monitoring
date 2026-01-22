# Real-Time Analysis Quick Start

## Features Added

✅ **Real-Time Video Analysis** with live preview  
✅ **YOLO Detection Boxes** on people in video  
✅ **Live Metrics**: Person count, risk score, confidence, time  
✅ **Progress Bar** showing analysis completion  
✅ **Risk Level Indicators** (NONE/LOW/MEDIUM/HIGH)  
✅ **Primary Cause Display** with explanations  

## How to Use

### 1. Start Backend
```bash
cd backend
pip install -r requirements.txt
pip install ultralytics torch
uvicorn app.main:app --reload
```

### 2. Start Frontend
```bash
cd frontend
npm install
npm run dev
```

### 3. Upload Video with Real-Time Analysis

1. Go to User Dashboard
2. Select a video file (MP4, AVI, MOV, MKV)
3. ✅ **Check "Real-Time Analysis Mode"** checkbox
4. Click "Start Real-Time Analysis"
5. Watch live detection with:
   - Bounding boxes on detected people
   - Real-time risk assessment
   - Person counting
   - Progress tracking

## What You'll See

- **Detection Boxes**: Green boxes around each detected person
- **Person Count**: Live count of people in frame
- **Risk Score**: 0-100 scale
- **Risk Level**: Color-coded (NONE/LOW/MEDIUM/HIGH)
- **Confidence**: Model confidence in risk assessment
- **Primary Cause**: Explanation of risk factors
- **Progress**: % completion of video analysis

## Technical Details

- **Model**: YOLOv8n (fast, accurate)
- **Processing**: ~10 FPS
- **Streaming**: Server-Sent Events (SSE)
- **Detection**: Person tracking with ByteTracker
- **Risk Engine**: Multi-signal temporal reasoning

## Comparison: Normal vs Real-Time Mode

**Normal Mode**:
- Process entire video
- Get final results
- No live preview

**Real-Time Mode** ✨:
- See each frame being processed
- Watch detections live
- Track progress in real-time
- Stop anytime

## Requirements

- Python 3.9+
- Node.js 18+
- 4GB RAM minimum
- GPU recommended (optional)

Enjoy your advanced crowd monitoring system! 🎉
