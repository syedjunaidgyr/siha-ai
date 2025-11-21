# YourCare AI Service - Python Implementation

Python-based AI microservice for vital signs detection from face analysis. This replaces the Node.js implementation with better ML libraries and improved accuracy.

## Features

- **Better Face Detection**: Uses MediaPipe (Google's face detection) instead of TensorFlow.js
- **Improved Image Processing**: OpenCV and Pillow for robust image handling
- **Better Signal Processing**: NumPy and SciPy for accurate vital signs analysis
- **No JPEG Issues**: Proper image decoding handles all JPEG formats
- **Same API**: Compatible with existing mobile app (no changes needed)
- **Preventive Health Engine**: NEWS2 + WHO-guideline based lifestyle recommendations with fever / respiratory forecasting

## Installation

1. **Create virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Running the Service

### Development Mode:
```bash
python app.py
```

The service will start on port 3001 (or PORT environment variable).

### Production Mode:
```bash
gunicorn -w 4 -b 0.0.0.0:3001 app:app
```

## API Endpoints

### Health Check
```
GET /health
```

### Analyze Video Frames
```
POST /api/ai/analyze-video
Content-Type: application/json

{
  "frames": ["base64_image_1", "base64_image_2", ...],
  "sensorData": {
    "accelerometer": { "magnitude": 1.3 },
    "ambientLight": { "illuminance": 150 }
  },
  "save": false
}
```

### Analyze Single Image
```
POST /api/ai/analyze-image
Content-Type: multipart/form-data

image: <file>
```

### Preventive Health & Lifestyle Insights
```
POST /api/ai/preventive-health
Content-Type: application/json

{
  "metrics": [
    { "type": "heart_rate", "value": 78, "timestamp": "2024-11-18T10:00:00Z" },
    { "type": "blood_pressure_systolic", "value": 118, "timestamp": "2024-11-18T10:00:00Z" },
    { "type": "stress_level", "value": 62, "timestamp": "2024-11-18T09:00:00Z" }
  ],
  "lookbackDays": 14,
  "userProfile": {
    "age": 32,
    "weightKg": 70
  }
}
```

Response includes NEWS2 score, fever/cough probabilities, personalized lifestyle plan, safety flags, and top recommendations.

## Environment Variables

- `PORT`: Server port (default: 3001)

## Advantages over Node.js Version

1. **Better Face Detection**: MediaPipe is more accurate and faster
2. **No JPEG Issues**: OpenCV handles all JPEG formats correctly
3. **Better Signal Processing**: SciPy provides advanced signal processing
4. **More Accurate Algorithms**: Better implementation of PPG and vital signs analysis
5. **Easier ML Integration**: Python ecosystem is better for ML/AI

## Migration from Node.js

The Python service uses the same API endpoints and response format, so no changes are needed in the mobile app. Just update the AI service URL if needed.

## Troubleshooting

### MediaPipe Installation Issues
If MediaPipe fails to install, try:
```bash
pip install --upgrade pip
pip install mediapipe
```

### OpenCV Issues
If OpenCV fails, try:
```bash
pip install opencv-python-headless
```

### Port Already in Use
Change the port:
```bash
PORT=3002 python app.py
```

