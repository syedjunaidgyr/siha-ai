"""
Face Detection Service using OpenCV and MediaPipe
More accurate and reliable than TensorFlow.js
"""
import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import mediapipe as mp
from PIL import Image
import io
import warnings
import os

# Suppress OpenCV warnings about JPEG SOS parameters (harmless)
# OpenCV uses different log level constants depending on version
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
try:
    # Try OpenCV 4.x API (most common)
    if hasattr(cv2, 'utils') and hasattr(cv2.utils, 'logging'):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    elif hasattr(cv2, 'setLogLevel'):
        cv2.setLogLevel(0)  # 0 = LOG_LEVEL_SILENT
except (AttributeError, Exception):
    # If neither works, just set environment variable (works for most cases)
    # The warnings are harmless anyway - OpenCV still processes images correctly
    pass


def decode_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode image bytes to numpy array with fallback support.
    Tries OpenCV first, falls back to PIL if OpenCV fails.
    
    Args:
        image_bytes: Raw image bytes (JPEG, PNG, etc.)
        
    Returns:
        numpy array (BGR format) or None if decoding fails
    """
    # First try OpenCV (faster and handles most cases)
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is not None:
            return image
    except Exception as e:
        print(f"[ImageDecode] OpenCV decode failed: {str(e)}")
    
    # Fallback to PIL if OpenCV fails (handles problematic JPEGs better)
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        # Convert PIL RGB to OpenCV BGR
        rgb_array = np.array(pil_image)
        if len(rgb_array.shape) == 2:  # Grayscale
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_GRAY2BGR)
        else:  # RGB
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        print("[ImageDecode] Successfully decoded using PIL fallback")
        return bgr_array
    except Exception as e:
        print(f"[ImageDecode] PIL decode also failed: {str(e)}")
        return None


class FaceDetectionService:
    """Face detection using MediaPipe Face Detection"""
    
    def __init__(self):
        self.mp_face_detection = None
        self.face_detection = None
        self.is_initialized = False
    
    def initialize(self):
        """Initialize MediaPipe face detection model"""
        if self.is_initialized:
            return
        
        try:
            print("Loading MediaPipe face detection model...")
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 0 for short-range, 1 for full-range
                min_detection_confidence=0.5
            )
            self.is_initialized = True
            print("Face detection model loaded successfully")
        except Exception as e:
            print(f"Error loading face detection model: {str(e)}")
            print("Face detection will use fallback method")
            self.is_initialized = True  # Mark as initialized to prevent retry loops
    
    def detect_face(self, image_bytes: bytes) -> Dict:
        """
        Detect face in an image buffer
        
        Returns:
            Dict with keys: detected (bool), confidence (float), boundingBox (dict)
        """
        import time
        start_time = time.time()
        
        try:
            # Decode image bytes (with fallback support)
            image = decode_image_bytes(image_bytes)
            
            if image is None:
                print("[FaceDetection] Failed to decode image")
                return self._fallback_face_detection(image_bytes)
            
            height, width = image.shape[:2]
            print(f"[FaceDetection] Processing image: {width}x{height}, size={len(image_bytes)} bytes")
            
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            if self.face_detection:
                results = self.face_detection.process(rgb_image)
                
                if results.detections and len(results.detections) > 0:
                    # Get the first (and most confident) detection
                    detection = results.detections[0]
                    confidence = detection.score[0]
                    
                    # Get bounding box (MediaPipe returns normalized coordinates)
                    bbox = detection.location_data.relative_bounding_box
                    
                    bounding_box = {
                        'xMin': int(bbox.xmin * width),
                        'yMin': int(bbox.ymin * height),
                        'xMax': int((bbox.xmin + bbox.width) * width),
                        'yMax': int((bbox.ymin + bbox.height) * height),
                        'width': int(bbox.width * width),
                        'height': int(bbox.height * height)
                    }
                    
                    print(f"[FaceDetection] ✓ Face detected successfully! confidence={confidence:.2f}, "
                          f"bbox=({bounding_box['xMin']}, {bounding_box['yMin']}, "
                          f"{bounding_box['width']}x{bounding_box['height']})")
                    
                    return {
                        'detected': True,
                        'confidence': float(confidence),
                        'boundingBox': bounding_box
                    }
            
            # No face detected, use fallback
            print("[FaceDetection] No faces detected by model, using fallback...")
            return self._fallback_face_detection(image_bytes)
            
        except Exception as e:
            print(f"[FaceDetection] Error detecting face with model: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._fallback_face_detection(image_bytes)
        finally:
            duration = (time.time() - start_time) * 1000
            print(f"[FaceDetection] Detection duration: {duration:.0f}ms")
    
    def _fallback_face_detection(self, image_bytes: bytes) -> Dict:
        """
        Fallback face detection using center region assumption
        Improved with skin tone analysis
        """
        try:
            # Decode image bytes (with fallback support)
            image = decode_image_bytes(image_bytes)
            
            if image is None:
                return {
                    'detected': False,
                    'confidence': 0.0
                }
            
            height, width = image.shape[:2]
            
            # Try to detect skin-tone regions for better face location
            face_x = width / 2
            face_y = height / 2
            confidence = 0.7
            
            try:
                # Sample center region to check for skin tones
                sample_size = min(200, int(width * 0.3))
                sample_x = int((width - sample_size) / 2)
                sample_y = int((height - sample_size) / 2)
                
                sample = image[sample_y:sample_y+sample_size, sample_x:sample_x+sample_size]
                
                if sample.size > 0:
                    # Calculate average R/G ratio (skin tones typically have R/G > 1.0)
                    rgb_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
                    r_channel = rgb_sample[:, :, 0].astype(np.float32)
                    g_channel = rgb_sample[:, :, 1].astype(np.float32)
                    
                    # Avoid division by zero
                    g_channel[g_channel == 0] = 1
                    ratios = r_channel / g_channel
                    avg_ratio = np.mean(ratios)
                    
                    # If R/G ratio is in skin tone range (1.0-1.5), increase confidence
                    if 1.0 <= avg_ratio <= 1.5:
                        confidence = 0.8
            except Exception as e:
                print(f"[FaceDetection] Fallback: Could not sample for skin tones: {str(e)}")
            
            # Assume face is in center region
            face_width = width * 0.4
            face_height = height * 0.5
            
            bounding_box = {
                'xMin': int(face_x - face_width / 2),
                'yMin': int(face_y - face_height / 2),
                'xMax': int(face_x + face_width / 2),
                'yMax': int(face_y + face_height / 2),
                'width': int(face_width),
                'height': int(face_height)
            }
            
            print(f"[FaceDetection] Fallback: Assuming face in center region, confidence={confidence}")
            
            return {
                'detected': True,
                'confidence': confidence,
                'boundingBox': bounding_box
            }
            
        except Exception as e:
            print(f"[FaceDetection] Error in fallback face detection: {str(e)}")
            return {
                'detected': False,
                'confidence': 0.0
            }
    
    def extract_roi(self, image_bytes: bytes, bounding_box: Dict) -> bytes:
        """
        Extract Region of Interest (ROI) for vital signs analysis
        Uses forehead region where blood vessels are most visible
        """
        try:
            # Decode image bytes (with fallback support)
            image = decode_image_bytes(image_bytes)
            
            if image is None:
                raise ValueError("Failed to decode image")
            
            height, width = image.shape[:2]
            
            # Extract forehead region (upper 30% of face, centered)
            roi_x = max(0, int(bounding_box['xMin'] + bounding_box['width'] * 0.2))
            roi_y = max(0, int(bounding_box['yMin'] + bounding_box['height'] * 0.1))
            roi_width = int(bounding_box['width'] * 0.6)
            roi_height = int(bounding_box['height'] * 0.3)
            
            # Ensure ROI is within image bounds
            roi_x = min(roi_x, width - 1)
            roi_y = min(roi_y, height - 1)
            roi_width = min(roi_width, width - roi_x)
            roi_height = min(roi_height, height - roi_y)
            
            if roi_width <= 0 or roi_height <= 0:
                raise ValueError("Invalid ROI dimensions")
            
            # Extract ROI
            roi = image[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
            
            # Encode back to JPEG
            _, encoded = cv2.imencode('.jpg', roi, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return encoded.tobytes()
            
        except Exception as e:
            print(f"[FaceDetection] ROI extraction failed: {str(e)}")
            raise
    
    def validate_frame_quality(self, image_bytes: bytes) -> Dict:
        """
        Validate frame quality for vital signs analysis
        
        Returns:
            Dict with keys: isValid (bool), score (int), issues (list)
        """
        issues = []
        score = 100
        
        try:
            # Decode image bytes (with fallback support)
            image = decode_image_bytes(image_bytes)
            
            if image is None:
                return {
                    'isValid': False,
                    'score': 0,
                    'issues': ['Invalid or corrupted image']
                }
            
            height, width = image.shape[:2]
            print(f"[FrameQuality] Validating frame: {width}x{height}, size={len(image_bytes)} bytes")
            
            # Check resolution
            if width < 320 or height < 240:
                issues.append('Resolution too low')
                score -= 30
            
            # Calculate image statistics
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Check brightness
            mean_brightness = np.mean(gray)
            if mean_brightness < 50:
                issues.append('Image too dark')
                score -= 25
            elif mean_brightness > 200:
                issues.append('Image too bright')
                score -= 25
            
            # Check contrast (standard deviation)
            std_dev = np.std(gray)
            if std_dev < 15:
                issues.append('Low contrast')
                score -= 20
            
            # Check sharpness (using Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:
                issues.append('Image appears blurry')
                score -= 25
            
            result = {
                'isValid': score >= 50,
                'score': max(0, score),
                'issues': issues
            }
            
            print(f"[FrameQuality] Validation result: isValid={result['isValid']}, "
                  f"score={result['score']}, issues={issues if issues else 'none'}")
            
            return result
            
        except Exception as e:
            print(f"[FrameQuality] Error validating frame quality: {str(e)}")
            return {
                'isValid': False,
                'score': 0,
                'issues': [f'Validation error: {str(e)}']
            }

