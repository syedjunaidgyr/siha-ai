"""
YourCare AI Analysis Service - Python Implementation
Flask-based microservice for vital signs detection from face analysis
"""
import os
import base64
import io
import time
import hashlib
from datetime import datetime
from typing import List, Dict, Optional
import warnings

# Suppress OpenCV warnings about JPEG SOS parameters (harmless)
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np

from services.face_detection import FaceDetectionService
from services.vital_signs import VitalSignsAnalysisService
from services.cache import CacheService
from services.preventive_health import PreventiveHealthInsightsService

app = Flask(__name__)
CORS(app)

# Increase max content length to handle large video payloads (50MB)
# This allows the 40MB+ payloads we're seeing
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Initialize services
face_detection_service = FaceDetectionService()
vital_signs_service = VitalSignsAnalysisService()
cache_service = CacheService()
preventive_health_service = PreventiveHealthInsightsService()

# Set face detection service reference in vital signs service
vital_signs_service.set_face_detection_service(face_detection_service)

# Service initialization flag
services_initialized = False


def initialize_services():
    """Initialize AI services on startup"""
    global services_initialized
    if not services_initialized:
        print("Initializing AI services...")
        face_detection_service.initialize()
        cache_service.initialize()
        preventive_health_service.initialize()
        services_initialized = True
        print("AI services initialized successfully")


@app.before_request
def before_request():
    """Initialize services before first request"""
    if not services_initialized:
        initialize_services()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'ai-analysis-python',
        'modelsLoaded': services_initialized,
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/api/ai/analyze-video', methods=['POST'])
def analyze_video():
    """Analyze multiple frames (video sequence) for vital signs"""
    try:
        start_time = time.time()
        
        # Log request size for debugging
        content_length = request.content_length
        if content_length:
            content_length_mb = content_length / (1024 * 1024)
            print(f"[AI Route] Request received: Content-Length={content_length_mb:.2f} MB")
        
        # Get frames from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        frames_base64 = data.get('frames', [])
        if not frames_base64 or not isinstance(frames_base64, list):
            return jsonify({'error': 'No frames provided'}), 400
        
        # Warn if too many frames (may cause memory issues)
        if len(frames_base64) > 20:
            print(f"[AI Route] WARNING: Large number of frames ({len(frames_base64)}). This may cause memory or timeout issues.")
        
        # Get sensor data if provided (optional)
        sensor_data = data.get('sensorData')
        if sensor_data:
            print(f"[AI Route] Received sensor data: motion={sensor_data.get('accelerometer') is not None}, "
                  f"proximity={sensor_data.get('proximity') is not None}, "
                  f"light={sensor_data.get('ambientLight') is not None}")
        
        print(f"[AI Route] Received analyze-video request: {len(frames_base64)} frames")
        
        # Convert base64 strings to image buffers
        frames = []
        invalid_frames = []
        for i, frame_str in enumerate(frames_base64):
            try:
                # Remove data URL prefix if present
                if ',' in frame_str:
                    frame_str = frame_str.split(',', 1)[1]
                
                # Decode base64
                frame_bytes = base64.b64decode(frame_str)
                
                # Validate that we got valid bytes
                if not frame_bytes or len(frame_bytes) < 100:  # Minimum reasonable JPEG size
                    print(f"[AI Route] Frame {i + 1}: Invalid frame size ({len(frame_bytes) if frame_bytes else 0} bytes)")
                    invalid_frames.append(i + 1)
                    continue
                
                # Validate JPEG header (should start with FF D8 FF)
                if frame_bytes[:3] != b'\xff\xd8\xff':
                    print(f"[AI Route] Frame {i + 1}: Invalid JPEG header")
                    invalid_frames.append(i + 1)
                    continue
                
                frames.append(frame_bytes)
                
                if i < 3:
                    print(f"[AI Route] Frame {i + 1}: base64 length={len(frame_str)}, buffer size={len(frame_bytes)} bytes")
            except Exception as e:
                print(f"[AI Route] Error parsing frame {i + 1}: {str(e)}")
                invalid_frames.append(i + 1)
                continue
        
        if invalid_frames:
            print(f"[AI Route] WARNING: {len(invalid_frames)} invalid frames skipped: {invalid_frames}")
        
        if len(frames) == 0:
            return jsonify({'error': 'No valid frames found after decoding'}), 400
        
        print(f"[AI Route] Processing {len(frames)} frames, first frame size: {len(frames[0])} bytes")
        
        # Check cache
        cache_key = cache_service.generate_key(frames[0])
        cached = cache_service.get(cache_key)
        if cached:
            print("[AI Route] Returning cached result")
            return jsonify({
                'success': True,
                'result': cached,
                'cached': True
            })
        
        print("[AI Route] Cache miss, starting analysis...")
        analysis_start_time = time.time()
        
        try:
            # Analyze the frames (with optional sensor data for quality adjustment)
            result = vital_signs_service.analyze_video_frames(frames, sensor_data=sensor_data)
            
            analysis_duration = (time.time() - analysis_start_time) * 1000  # Convert to ms
            print(f"[AI Route] Analysis completed in {analysis_duration:.0f}ms")
        except MemoryError as me:
            print(f"[AI Route] Memory error during analysis: {str(me)}")
            # Clean up frames to free memory
            del frames
            import gc
            gc.collect()
            return jsonify({
                'error': 'Out of memory',
                'message': 'The video frames are too large to process. Please reduce the number of frames or image resolution.'
            }), 413
        except Exception as analysis_error:
            print(f"[AI Route] Error during video analysis: {str(analysis_error)}")
            import traceback
            traceback.print_exc()
            # Clean up frames to free memory
            del frames
            import gc
            gc.collect()
            raise  # Re-raise to be caught by outer exception handler
        
        # Cache result (5 minutes TTL)
        cache_service.set(cache_key, result, 300)
        print("[AI Route] Result cached")
        
        print(f"[AI Route] Sending response: faceDetected={result['faceDetected']}, "
              f"totalFrames={result.get('totalFrames', 0)}, hasVitals={bool(result.get('vitals', {}).get('heartRate'))}")
        
        return jsonify({
            'success': True,
            'result': result,
            'cached': False
        })
        
    except MemoryError as me:
        print(f"[AI Route] Memory error - payload too large: {str(me)}")
        return jsonify({
            'error': 'Payload too large',
            'message': 'The video frames are too large to process. Please reduce the number of frames or image resolution.'
        }), 413
    except Exception as e:
        error_msg = str(e)
        print(f"[AI Route] AI video analysis error: {error_msg}")
        import traceback
        traceback.print_exc()
        
        # Provide more helpful error messages
        if 'timeout' in error_msg.lower() or 'timed out' in error_msg.lower():
            return jsonify({
                'error': 'Analysis timed out',
                'message': 'The analysis took too long. Try reducing the number of frames.'
            }), 504
        elif 'memory' in error_msg.lower() or 'out of memory' in error_msg.lower():
            return jsonify({
                'error': 'Out of memory',
                'message': 'The payload is too large. Please reduce the number of frames.'
            }), 413
        
        return jsonify({
            'error': 'Analysis failed',
            'message': error_msg
        }), 500


@app.route('/api/ai/analyze-image', methods=['POST'])
def analyze_image():
    """Analyze a single image frame for vital signs"""
    try:
        # Check if file was uploaded
        if 'image' in request.files:
            file = request.files['image']
            image_bytes = file.read()
        else:
            # Try to get from JSON
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({'error': 'No image provided'}), 400
            
            image_str = data['image']
            if ',' in image_str:
                image_str = image_str.split(',', 1)[1]
            image_bytes = base64.b64decode(image_str)
        
        # Check cache
        cache_key = cache_service.generate_key(image_bytes)
        cached = cache_service.get(cache_key)
        if cached:
            return jsonify({
                'success': True,
                'result': cached,
                'cached': True
            })
        
        # Analyze the image
        result = vital_signs_service.analyze_image_frame(image_bytes)
        
        # Cache result (5 minutes TTL)
        cache_service.set(cache_key, result, 300)
        
        return jsonify({
            'success': True,
            'result': result,
            'cached': False
        })
        
    except Exception as e:
        print(f"AI analysis error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Analysis failed',
            'message': str(e)
        }), 500


@app.route('/api/ai/preventive-health', methods=['POST'])
def preventive_health():
    """Generate preventive health & lifestyle insights from stored metrics"""
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({'error': 'Request body is required'}), 400

        metrics = payload.get('metrics')
        if not metrics or not isinstance(metrics, list):
            return jsonify({'error': 'metrics must be a non-empty list'}), 400

        user_profile = payload.get('userProfile')
        try:
            lookback_days = int(payload.get('lookbackDays', 14))
        except (TypeError, ValueError):
            lookback_days = 14

        result = preventive_health_service.generate_insights(
            metrics=metrics,
            user_profile=user_profile,
            lookback_days=lookback_days
        )

        return jsonify({
            'success': True,
            'result': result
        })
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print(f"[AI Route] Preventive health error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Preventive insight generation failed',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    # Initialize services
    initialize_services()
    
    # Get port from environment or default to 3001
    port = int(os.environ.get('PORT', 3001))
    
    # Check if running in production (use gunicorn) or development
    is_production = os.environ.get('FLASK_ENV') == 'production' or os.environ.get('ENV') == 'production'
    
    if is_production:
        print(f"Starting YourCare AI Service (Python) in PRODUCTION mode on port {port}...")
        print("NOTE: Use gunicorn for production: gunicorn -w 2 -b 0.0.0.0:3001 --timeout 300 app:app")
        # In production, don't use debug mode
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    else:
        print(f"Starting YourCare AI Service (Python) in DEVELOPMENT mode on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)

