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
import tempfile

# Suppress OpenCV warnings about JPEG SOS parameters (harmless)
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2

from services.face_detection import FaceDetectionService
from services.vital_signs import VitalSignsAnalysisService
from services.cache import CacheService
from services.preventive_health import PreventiveHealthInsightsService

app = Flask(__name__)
CORS(app)

# Increase max content length to handle large video payloads (100MB)
# 30-second videos at high quality can be 60-80MB
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

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
        
        # Limit frame count to prevent memory issues
        MAX_FRAMES = 12
        if len(frames_base64) > MAX_FRAMES:
            print(f"[AI Route] WARNING: Frame count ({len(frames_base64)}) exceeds maximum ({MAX_FRAMES}). Using first {MAX_FRAMES} frames.")
            frames_base64 = frames_base64[:MAX_FRAMES]
        
        # Warn if payload is large
        if len(frames_base64) > 10:
            print(f"[AI Route] WARNING: Large number of frames ({len(frames_base64)}). Processing with memory optimization.")
        
        # Get sensor data if provided (optional)
        sensor_data = data.get('sensorData')
        if sensor_data:
            print(f"[AI Route] Received sensor data: motion={sensor_data.get('accelerometer') is not None}, "
                  f"proximity={sensor_data.get('proximity') is not None}, "
                  f"light={sensor_data.get('ambientLight') is not None}")
        
        # Get user profile if provided (optional, for calibration and personalized baselines)
        user_profile = data.get('userProfile')
        if user_profile:
            print(f"[AI Route] Received user profile: age={user_profile.get('age')}, gender={user_profile.get('gender')}")
        
        print(f"[AI Route] Received analyze-video request: {len(frames_base64)} frames")
        
        # Convert base64 strings to image buffers with memory optimization
        # Resize large images to reduce memory usage
        MAX_IMAGE_WIDTH = 1920  # Resize if wider than this
        MAX_IMAGE_HEIGHT = 1080  # Resize if taller than this
        MAX_FRAME_SIZE_MB = 2  # Warn if individual frame > 2MB
        
        frames = []
        invalid_frames = []
        import gc
        
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
                    del frame_bytes
                    continue
                
                # Check if it's JPEG format (starts with FF D8 FF)
                # If not JPEG, the frame might be raw YUV/RGB data from frame processor
                # decode_image_bytes can handle various formats, so we'll try to decode it
                is_jpeg = len(frame_bytes) >= 3 and frame_bytes[:3] == b'\xff\xd8\xff'
                if not is_jpeg:
                    # Frame is likely raw pixel data (YUV, RGB, etc.) from frame processor
                    # decode_image_bytes will attempt to decode it
                    print(f"[AI Route] Frame {i + 1}: Not JPEG format (likely raw pixel data), will attempt decode")
                    # Continue - decode_image_bytes will try to handle it
                
                # Check frame size and resize if too large
                frame_size_mb = len(frame_bytes) / (1024 * 1024)
                if frame_size_mb > MAX_FRAME_SIZE_MB:
                    print(f"[AI Route] Frame {i + 1}: Large frame ({frame_size_mb:.2f} MB), resizing to reduce memory usage")
                    try:
                        # Decode image for resizing
                        image = Image.open(io.BytesIO(frame_bytes))
                        width, height = image.size
                        
                        # Resize if larger than max dimensions
                        if width > MAX_IMAGE_WIDTH or height > MAX_IMAGE_HEIGHT:
                            # Calculate new dimensions maintaining aspect ratio
                            ratio = min(MAX_IMAGE_WIDTH / width, MAX_IMAGE_HEIGHT / height)
                            new_width = int(width * ratio)
                            new_height = int(height * ratio)
                            
                            # Resize image
                            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                            
                            # Re-encode as JPEG
                            output = io.BytesIO()
                            image.save(output, format='JPEG', quality=85, optimize=True)
                            frame_bytes = output.getvalue()
                            print(f"[AI Route] Frame {i + 1}: Resized from {width}x{height} to {new_width}x{new_height}, new size: {len(frame_bytes) / (1024 * 1024):.2f} MB")
                            
                            del image
                            del output
                    except Exception as resize_error:
                        print(f"[AI Route] Frame {i + 1}: Error resizing frame, using original: {str(resize_error)}")
                        # Use original if resize fails
                
                frames.append(frame_bytes)
                del frame_bytes  # Remove reference, frame is in list now
                
                if i < 3:
                    print(f"[AI Route] Frame {i + 1}: base64 length={len(frame_str)}, buffer size={len(frames[-1])} bytes")
                
                # Periodic garbage collection for large payloads
                if i > 0 and i % 5 == 0:
                    gc.collect()
                    
            except Exception as e:
                print(f"[AI Route] Error parsing frame {i + 1}: {str(e)}")
                invalid_frames.append(i + 1)
                if 'frame_bytes' in locals():
                    del frame_bytes
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
        
        # Force garbage collection before processing to free up memory
        import gc
        gc.collect()
        
        try:
            # Analyze the frames (with optional sensor data and user profile for calibration)
            result = vital_signs_service.analyze_video_frames(frames, sensor_data=sensor_data, user_profile=user_profile)
            
            analysis_duration = (time.time() - analysis_start_time) * 1000  # Convert to ms
            print(f"[AI Route] Analysis completed in {analysis_duration:.0f}ms")
            
            # Clean up frames after processing to free memory
            # Explicitly delete frames and clear list
            for frame in frames:
                del frame
            frames.clear()
            del frames
            gc.collect()
        except MemoryError as me:
            print(f"[AI Route] Memory error during analysis: {str(me)}")
            # Clean up frames to free memory
            del frames
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
            gc.collect()
            raise  # Re-raise to be caught by outer exception handler
        
        # Validate result structure
        if not isinstance(result, dict):
            print(f"[AI Route] ERROR: Result is not a dictionary: {type(result)}")
            raise ValueError(f"Invalid result type from analysis: {type(result)}")
        
        # Cache result (5 minutes TTL)
        cache_service.set(cache_key, result, 300)
        print("[AI Route] Result cached")
        
        # Safely access result fields
        face_detected = result.get('faceDetected', False)
        total_frames = result.get('totalFrames', 0)
        has_vitals = bool(result.get('vitals', {}).get('heartRate'))
        print(f"[AI Route] Sending response: faceDetected={face_detected}, "
              f"totalFrames={total_frames}, hasVitals={has_vitals}")
        
        # Ensure result has required fields
        if 'faceDetected' not in result:
            print("[AI Route] WARNING: Result missing 'faceDetected' key, adding default value")
            result['faceDetected'] = face_detected
        
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


@app.route('/api/ai/analyze-video-file', methods=['POST'])
def analyze_video_file():
    """Analyze a video file for vital signs (video-based pipeline)"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        payload = request.get_json()
        
        if 'video' not in payload:
            return jsonify({'error': 'No video data provided'}), 400
        
        video_base64 = payload.get('video')
        mime_type = payload.get('mimeType', 'video/mp4')
        sensor_data = payload.get('sensorData')
        user_profile = payload.get('userProfile')
        
        print(f"[AI Route] Video file received: {len(video_base64)} base64 chars, type: {mime_type}")
        
        # Decode video from base64
        try:
            video_bytes = base64.b64decode(video_base64)
            print(f"[AI Route] Decoded video: {len(video_bytes)} bytes")
        except Exception as e:
            print(f"[AI Route] Error decoding video: {str(e)}")
            return jsonify({'error': f'Invalid video data: {str(e)}'}), 400
        
        # Save video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(video_bytes)
            temp_video_path = temp_video.name
        
        try:
            # Extract frames from video using OpenCV
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                return jsonify({'error': 'Failed to open video file'}), 400
            
            frames = []
            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Default to 30 FPS if unknown
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"[AI Route] Video properties: {total_frames} frames, {fps} FPS")
            
            # Sample frames at ~10 FPS for analysis (30 FPS video -> sample every 3 frames)
            # This gives us ~100 frames for a 30-second video, which is enough for accurate HR detection
            # We'll process all of them since we have the actual video FPS
            frame_interval = max(1, int(fps / 10))  # Sample at ~10 FPS (every 3 frames for 30 FPS video)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames at the specified interval
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert to PIL Image then to bytes
                    pil_image = Image.fromarray(frame_rgb)
                    img_byte_arr = io.BytesIO()
                    pil_image.save(img_byte_arr, format='JPEG', quality=85)
                    img_byte_arr = img_byte_arr.getvalue()
                    frames.append(img_byte_arr)
                
                frame_count += 1
            
            cap.release()
            
            if len(frames) == 0:
                return jsonify({'error': 'No frames extracted from video'}), 400
            
            print(f"[AI Route] Extracted {len(frames)} frames from video at ~{fps/frame_interval:.1f} FPS")
            
            # Analyze frames using existing vital signs service
            # Pass the actual video FPS so it can calculate correctly
            result = vital_signs_service.analyze_video_frames(
                frames, 
                sensor_data=sensor_data, 
                user_profile=user_profile,
                video_fps=fps / frame_interval  # Actual FPS of extracted frames
            )
            
            # Ensure result has required fields
            if not isinstance(result, dict):
                result = {'faceDetected': False, 'totalFrames': len(frames), 'vitals': {}}
            if 'faceDetected' not in result:
                result['faceDetected'] = result.get('validFrames', 0) > 0
            if 'totalFrames' not in result:
                result['totalFrames'] = len(frames)
            
            return jsonify({
                'success': True,
                'result': result
            })
            
        finally:
            # Clean up temporary video file
            try:
                os.unlink(temp_video_path)
            except:
                pass
        
    except Exception as e:
        error_msg = str(e)
        print(f"[AI Route] Video file analysis error: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Video analysis failed',
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
    import time
    start_time = time.time()
    
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({'error': 'Request body is required'}), 400

        metrics = payload.get('metrics')
        # Allow empty metrics list - AI will generate recommendations from user profile
        if metrics is None:
            metrics = []
        if not isinstance(metrics, list):
            return jsonify({'error': 'metrics must be a list'}), 400

        print(f"[AI Route] Preventive health request: {len(metrics)} metrics, lookback_days={payload.get('lookbackDays', 14)}")
        
        user_profile = payload.get('userProfile')
        try:
            lookback_days = int(payload.get('lookbackDays', 14))
        except (TypeError, ValueError):
            lookback_days = 14

        print(f"[AI Route] Starting insights generation...")
        result = preventive_health_service.generate_insights(
            metrics=metrics,
            user_profile=user_profile,
            lookback_days=lookback_days
        )
        
        elapsed = time.time() - start_time
        print(f"[AI Route] Insights generation completed in {elapsed:.2f}s")

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

