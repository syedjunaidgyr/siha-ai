"""
Vital Signs Analysis Service using PPG and signal processing
Improved algorithms for better accuracy with advanced signal processing
"""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import signal
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
import time
from PIL import Image
import io


def decode_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode image bytes to numpy array with fallback support.
    Tries OpenCV first, falls back to PIL if OpenCV fails.
    For large images (>1MB), uses PIL directly to avoid memory issues.
    
    Args:
        image_bytes: Raw image bytes (JPEG, PNG, etc.)
        
    Returns:
        numpy array (BGR format) or None if decoding fails
    """
    import sys
    import contextlib
    
    # For large images, use PIL directly to avoid OpenCV memory issues
    if len(image_bytes) > 1024 * 1024:  # > 1MB
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            pil_image.load()
            rgb_array = np.array(pil_image)
            if len(rgb_array.shape) == 2:  # Grayscale
                bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_GRAY2BGR)
            else:  # RGB
                bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            return bgr_array
        except Exception as e:
            print(f"[ImageDecode] PIL decode failed for large image: {str(e)}")
            return None
    
    # Suppress stderr to prevent OpenCV from printing "Invalid SOS parameters" error
    # This error is harmless and we'll use PIL fallback anyway
    with contextlib.redirect_stderr(io.StringIO()):
        # First try OpenCV (faster and handles most cases)
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is not None:
                return image
        except Exception:
            # Silently catch - we'll try PIL
            pass
    
    # Fallback to PIL if OpenCV fails (handles problematic JPEGs better)
    # PIL is more tolerant of JPEG encoding issues
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        # Ensure image is loaded (lazy loading)
        pil_image.load()
        # Convert PIL RGB to OpenCV BGR
        rgb_array = np.array(pil_image)
        if len(rgb_array.shape) == 2:  # Grayscale
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_GRAY2BGR)
        else:  # RGB
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return bgr_array
    except Exception as e:
        print(f"[ImageDecode] PIL decode failed: {str(e)}")
        return None


class VitalSignsAnalysisService:
    """Analyze vital signs from face video frames"""
    
    def __init__(self):
        self.face_detection_service = None  # Will be set from app.py
    
    def set_face_detection_service(self, service):
        """Set face detection service reference"""
        self.face_detection_service = service
    
    def analyze_video_frames(self, frames: List[bytes], sensor_data: Optional[Dict] = None, user_profile: Optional[Dict] = None, video_fps: Optional[float] = None) -> Dict:
        """
        Analyze multiple frames for vital signs using temporal signal analysis
        This is the proper way to do PPG - analyze signal over time
        
        Args:
            frames: List of image bytes
            sensor_data: Optional sensor data (accelerometer, proximity, ambient light)
            user_profile: Optional user profile (age, gender, height, weight) for calibration
            
        Returns:
            Dict with analysis results
        """
        import gc
        
        start_time = time.time()
        total_frame_count = len(frames)
        print(f"[VitalSigns] Starting video frame analysis: {total_frame_count} frames")
        
        if not frames or len(frames) == 0:
            raise ValueError("No frames provided")
        
        # First pass: detect faces and extract ROIs from all frames
        # Process frames incrementally to minimize memory usage
        rois = []
        face_detected_count = 0
        total_quality_score = 0
        bounding_boxes = []
        previous_frame_gray = None  # For motion detection
        
        # Process all frames if we have video FPS (from video file), otherwise limit for memory
        # Video files have known FPS, so we can process more frames accurately
        if video_fps is not None and video_fps >= 5.0:
            # For video files with known FPS, process all frames (up to 150 for safety)
            MAX_FRAMES_TO_PROCESS = min(150, len(frames))
            frames_to_process = frames[:MAX_FRAMES_TO_PROCESS] if len(frames) > MAX_FRAMES_TO_PROCESS else frames
            frames_processed = len(frames_to_process)
            if len(frames) > MAX_FRAMES_TO_PROCESS:
                print(f"[VitalSigns] Processing {MAX_FRAMES_TO_PROCESS} frames from {len(frames)} (video FPS: {video_fps:.1f})")
        else:
            # For frame-based uploads without known FPS, limit to 12 for memory safety
            MAX_FRAMES_TO_PROCESS = 12
            frames_to_process = frames[:MAX_FRAMES_TO_PROCESS] if len(frames) > MAX_FRAMES_TO_PROCESS else frames
            frames_processed = len(frames_to_process)
            if len(frames) > MAX_FRAMES_TO_PROCESS:
                print(f"[VitalSigns] Limiting frame processing from {len(frames)} to {MAX_FRAMES_TO_PROCESS} frames for memory efficiency")
        
        # Motion detection threshold (normalized flow magnitude)
        MOTION_THRESHOLD = 2.0  # Discard frames with motion above this threshold
        
        for i, frame_bytes in enumerate(frames_to_process):
            try:
                # Validate frame quality
                quality_check = self.face_detection_service.validate_frame_quality(frame_bytes)
                if not quality_check['isValid'] and quality_check['score'] == 0:
                    # Explicitly delete frame_bytes reference to help GC
                    del frame_bytes
                    continue
                
                total_quality_score += quality_check['score']
                
                # Detect face
                face_result = self.face_detection_service.detect_face(frame_bytes)
                if not face_result.get('detected'):
                    del frame_bytes
                    continue
                
                # Motion detection using optical flow (if previous frame available)
                if previous_frame_gray is not None and i > 0:
                    try:
                        # Decode current frame for motion detection
                        current_image = decode_image_bytes(frame_bytes)
                        if current_image is not None:
                            current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
                            
                            # Use helper method for motion detection
                            has_motion, motion_magnitude = self._detect_frame_motion(previous_frame_gray, current_gray)
                            
                            if has_motion:
                                print(f"[VitalSigns] Frame {i+1} discarded due to motion: {motion_magnitude:.2f} > {MOTION_THRESHOLD}")
                                # Update previous frame but don't process current frame
                                previous_frame_gray = current_gray.copy()
                                del current_image
                                del frame_bytes
                                continue
                            
                            if motion_magnitude > 0:
                                print(f"[VitalSigns] Frame {i+1} motion level: {motion_magnitude:.2f} (OK)")
                            
                            # Update previous frame for next iteration
                            previous_frame_gray = current_gray.copy()
                            del current_image
                    except Exception as motion_error:
                        print(f"[VitalSigns] Motion detection failed for frame {i+1}: {str(motion_error)}")
                        # Continue processing if motion detection fails
                        # Try to initialize previous frame if not set
                        if previous_frame_gray is None:
                            try:
                                current_image = decode_image_bytes(frame_bytes)
                                if current_image is not None:
                                    previous_frame_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
                                    del current_image
                            except:
                                pass
                
                # If first frame, initialize previous frame for motion detection
                if previous_frame_gray is None:
                    try:
                        current_image = decode_image_bytes(frame_bytes)
                        if current_image is not None:
                            previous_frame_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
                            del current_image
                    except:
                        pass
                
                face_detected_count += 1
                bounding_boxes.append(face_result['boundingBox'])
                
                # Extract ROI (this creates a new smaller buffer)
                try:
                    roi_bytes = self.face_detection_service.extract_roi(
                        frame_bytes,
                        face_result['boundingBox']
                    )
                    rois.append(roi_bytes)
                    # Delete original frame after extracting ROI to free memory
                    del frame_bytes
                    # Don't delete roi_bytes - it's needed in the rois list
                except Exception as e:
                    print(f"[VitalSigns] ROI extraction failed for frame {i+1}: {str(e)}")
                    del frame_bytes
                    continue
                
                # Aggressive garbage collection every 3 frames to manage memory
                if (i + 1) % 3 == 0:
                    gc.collect()
                    
            except MemoryError as me:
                print(f"[VitalSigns] Memory error processing frame {i+1}: {str(me)}")
                gc.collect()
                # Stop processing if we hit memory limits
                break
            except Exception as e:
                print(f"[VitalSigns] Error processing frame {i+1}: {str(e)}")
                continue
            finally:
                # Ensure frame_bytes is deleted even if exception occurs
                if 'frame_bytes' in locals():
                    del frame_bytes
        
        # Clear frames list reference to help GC
        del frames_to_process
        del frames
        
        # Final aggressive garbage collection after processing all frames
        gc.collect()
        
        # Minimum frames needed for temporal analysis (reduced from 10 to 5 for flexibility)
        MIN_FRAMES_FOR_TEMPORAL = 5
        if len(rois) < MIN_FRAMES_FOR_TEMPORAL:
            print(f"[VitalSigns] Not enough valid frames ({len(rois)}), using fallback")
            # Use safe fallback that doesn't require frames array
            return self._fallback_analysis_safe(face_detected_count, total_quality_score, total_frame_count)
        
        print(f"[VitalSigns] Extracted {len(rois)} valid ROIs for temporal analysis")
        
        # Use overlapping sliding windows for short videos (< 10 frames) to improve accuracy
        SHORT_VIDEO_THRESHOLD = 10
        use_overlapping_windows = len(rois) < SHORT_VIDEO_THRESHOLD
        
        if use_overlapping_windows:
            print(f"[VitalSigns] Short video detected ({len(rois)} frames), using overlapping sliding windows")
            return self._analyze_with_overlapping_windows(rois, sensor_data, face_detected_count, total_quality_score, total_frame_count, user_profile)
        
        # Adjust quality score based on sensor data
        if sensor_data:
            sensor_quality_adjustment = self._adjust_quality_from_sensors(sensor_data)
            if sensor_quality_adjustment < 0:
                print(f"[VitalSigns] Sensor data indicates poor conditions, adjusting quality score by {sensor_quality_adjustment}")
                total_quality_score = max(0, total_quality_score + sensor_quality_adjustment)
        
        # Calculate actual frame rate (FPS) from number of frames
        # If video_fps is provided (from video file), use it directly
        if video_fps is not None and video_fps > 0:
            actual_fps = float(video_fps)
            estimated_duration = len(rois) / actual_fps if actual_fps > 0 else len(rois)
            print(f"[VitalSigns] Using video FPS: {actual_fps:.2f} Hz (from {len(rois)} frames over ~{estimated_duration:.1f}s)")
        else:
            # For frame-based uploads without known FPS, estimate from frame count
            # For mobile app: typically captures 20-30 frames over 30 seconds with 1s interval
            # takePhoto() takes 1-2 seconds per photo, so actual rate is ~0.8-1.0 FPS
            # Better estimation: assume 1 frame per second for photo capture mode
            if len(rois) >= 15:
                # If we have 15+ frames, assume they were captured over 20-30 seconds
                estimated_duration = max(20.0, len(rois) * 1.0)  # ~1.0 second per frame
            elif len(rois) >= 10:
                # If we have 10-14 frames, assume they were captured over 12-20 seconds
                estimated_duration = max(12.0, len(rois) * 1.0)  # ~1.0 second per frame
            else:
                # For fewer frames, assume similar rate (1 frame per second)
                estimated_duration = max(len(rois) * 1.0, 5.0)  # At least 5 seconds minimum
            
            actual_fps = len(rois) / estimated_duration if len(rois) > 0 else 1.0
            # Clamp FPS to realistic range (0.5-15 FPS for photo capture)
            # Minimum 0.5 FPS to handle very slow capture, but warn user
            actual_fps = np.clip(actual_fps, 0.5, 15.0)
        
        # Warn if FPS is too low for accurate heart rate detection (only for estimated FPS, not known video FPS)
        if video_fps is None:  # Only warn for estimated FPS
            if actual_fps < 1.0:
                print(f"[VitalSigns] WARNING: Very low FPS ({actual_fps:.2f} Hz) - heart rate detection will be inaccurate")
                print(f"[VitalSigns] Recommendation: Ensure stable capture, reduce movement, and hold device steady")
            elif actual_fps < 2.0:
                print(f"[VitalSigns] WARNING: Low FPS ({actual_fps:.2f} Hz) - heart rate detection may be inaccurate")
                print(f"[VitalSigns] Recommendation: Capture more frames or ensure consistent capture timing")
            else:
                print(f"[VitalSigns] Estimated FPS: {actual_fps:.2f} Hz (from {len(rois)} frames over ~{estimated_duration:.1f}s)")
        
        # Extract temporal signal from all ROIs with multi-ROI support
        try:
            # Extract signals from multiple regions for better accuracy, focusing on forehead
            signals = self._extract_multi_roi_signals(rois)
            
            if len(signals['red']) < 10:
                print(f"[VitalSigns] Not enough signals extracted: {len(signals['red'])}")
                return self._fallback_analysis_safe(face_detected_count, total_quality_score, total_frame_count)
            
            # Log raw signal statistics
            print(f"[VitalSigns] Raw signal stats - Red: mean={np.mean(signals['red']):.2f}, std={np.std(signals['red']):.2f}, range={np.max(signals['red'])-np.min(signals['red']):.2f}")
            print(f"[VitalSigns] Raw signal stats - Green: mean={np.mean(signals['green']):.2f}, std={np.std(signals['green']):.2f}, range={np.max(signals['green'])-np.min(signals['green']):.2f}")
            print(f"[VitalSigns] Raw signal stats - Blue: mean={np.mean(signals['blue']):.2f}, std={np.std(signals['blue']):.2f}, range={np.max(signals['blue'])-np.min(signals['blue']):.2f}")
            
            # Check signal quality before processing
            signal_quality = self._validate_signal_quality(signals)
            if not signal_quality['isValid']:
                print(f"[VitalSigns] Signal quality check failed: {signal_quality['reason']}")
                return self._fallback_analysis_safe(face_detected_count, total_quality_score, total_frame_count)
            
            # Advanced signal preprocessing with interpolation for low frame rates
            red_signal = self._preprocess_signal(signals['red'], sensor_data, fps=actual_fps)
            green_signal = self._preprocess_signal(signals['green'], sensor_data, fps=actual_fps)
            blue_signal = self._preprocess_signal(signals['blue'], sensor_data, fps=actual_fps)
            
            # Log processed signal statistics
            print(f"[VitalSigns] Processed signal stats - Red: mean={np.mean(red_signal):.2f}, std={np.std(red_signal):.2f}, range={np.max(red_signal)-np.min(red_signal):.2f}")
            
            # Calculate vital signs using ensemble methods with actual FPS
            heart_rate = self._calculate_heart_rate_ensemble(red_signal, green_signal, blue_signal, fps=actual_fps)
            print(f"[VitalSigns] Calculated heart rate: {heart_rate} BPM")
            
            respiratory_rate = self._calculate_respiratory_rate_ensemble(red_signal, fps=actual_fps)
            print(f"[VitalSigns] Calculated respiratory rate: {respiratory_rate} breaths/min")
            
            # For stress, temperature, and SpO2, use weighted average of multiple frames
            stress_level = self._calculate_stress_from_rois(rois[-10:])  # Last 10 frames for better accuracy
            print(f"[VitalSigns] Calculated stress level: {stress_level}")
            
            oxygen_saturation = self._calculate_oxygen_saturation_improved(rois[-10:], red_signal, green_signal, blue_signal, user_profile)
            print(f"[VitalSigns] Calculated SpO2: {oxygen_saturation}%")

            temperature = self._estimate_temperature_from_rois(rois[-10:], sensor_data, user_profile)
            print(f"[VitalSigns] Estimated temperature: {temperature:.2f}°C")

            # Estimate blood pressure based on heart rate, stress, and signal characteristics
            blood_pressure = self._estimate_blood_pressure(heart_rate, stress_level, red_signal)
            print(f"[VitalSigns] Calculated BP: {blood_pressure['systolic']}/{blood_pressure['diastolic']} mmHg")
            
            # Calculate signal quality metrics for enhanced confidence scoring
            signal_quality_metrics = {}
            try:
                if len(red_signal) >= 10:
                    # Calculate SNR and stability from processed signals
                    signal_quality_metrics = self._calculate_signal_quality_metrics(
                        red_signal, green_signal, blue_signal, motion_level=None
                    )
                    print(f"[VitalSigns] Signal quality - SNR: {signal_quality_metrics.get('snr', 0):.2f} dB, Stability: {signal_quality_metrics.get('stability', 0):.2f}")
            except Exception as sq_error:
                print(f"[VitalSigns] Signal quality calculation failed: {str(sq_error)}")
                signal_quality_metrics = {}
            
            # Calculate confidence with enhanced metrics
            avg_quality_score = total_quality_score / face_detected_count if face_detected_count > 0 else 0
            confidence = self._calculate_confidence(
                face_detected_count / total_frame_count if total_frame_count > 0 else 0,
                avg_quality_score,
                heart_rate,
                stress_level,
                oxygen_saturation,
                respiratory_rate,
                temperature,
                signal_quality=signal_quality_metrics if signal_quality_metrics else None
            )

        except Exception as e:
            print(f"[VitalSigns] Error in temporal analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._fallback_analysis_safe(face_detected_count, total_quality_score, total_frame_count)
        
        # If we reach here, confidence was already calculated above
        if 'confidence' not in locals():
            # Fallback confidence calculation if signals weren't available
            avg_quality_score = total_quality_score / face_detected_count if face_detected_count > 0 else 0
            confidence = self._calculate_confidence(
                face_detected_count / total_frame_count if total_frame_count > 0 else 0,
                avg_quality_score,
                heart_rate,
                stress_level,
                oxygen_saturation,
                respiratory_rate,
                temperature
            )
        
        duration = (time.time() - start_time) * 1000
        
        result = {
            'faceDetected': face_detected_count > 0,
            'validFrames': len(rois),
            'totalFrames': total_frame_count,
            'frameCount': total_frame_count,  # Alias for compatibility
            'vitals': {
                'heartRate': heart_rate,
                'stressLevel': stress_level,
                'oxygenSaturation': oxygen_saturation,
                'respiratoryRate': respiratory_rate,
                'temperature': temperature,
                'bloodPressure': blood_pressure,
                'confidence': confidence,
                'timestamp': time.time() * 1000  # milliseconds
            },
            'confidence': f"{confidence:.2f}",
            'avgQualityScore': f"{avg_quality_score:.1f}",
            'duration': f"{duration:.0f}ms"
        }
        
        print(f"[VitalSigns] ✓ Video analysis complete: {result}")
        return result
    
    def _fallback_analysis_safe(self, face_count: int, quality_score: float, total_frames: int) -> Dict:
        """Safe fallback analysis when not enough frames or memory issues"""
        avg_quality_score = quality_score / face_count if face_count > 0 else 0
        
        return {
            'faceDetected': face_count > 0,
            'validFrames': 0,
            'totalFrames': total_frames,
            'frameCount': total_frames,
            'vitals': {
                'heartRate': None,
                'stressLevel': None,
                'oxygenSaturation': None,
                'respiratoryRate': None,
                'temperature': None,
                'bloodPressure': {'systolic': None, 'diastolic': None},
                'confidence': 0.3,
                'timestamp': time.time() * 1000
            },
            'confidence': '0.30',
            'avgQualityScore': f"{avg_quality_score:.1f}"
        }
    
    def _fallback_analysis(self, frames: List[bytes], face_count: int, quality_score: float) -> Dict:
        """Fallback to per-frame analysis if temporal analysis fails"""
        results = []
        for i, frame_bytes in enumerate(frames[:10]):  # Analyze first 10 frames
            try:
                result = self.analyze_image_frame(frame_bytes)
                if result.get('faceDetected'):
                    results.append(result)
            except:
                continue
        
        if not results:
            return {
                'faceDetected': False,
                'validFrames': 0,
                'totalFrames': len(frames),
                'frameCount': len(frames),
                'vitals': {},
                'confidence': '0.00',
                'avgQualityScore': '0.0'
            }
        
        heart_rates = [r['vitals'].get('heartRate') for r in results if r.get('vitals', {}).get('heartRate')]
        stress_levels = [r['vitals'].get('stressLevel') for r in results if r.get('vitals', {}).get('stressLevel')]
        oxygen_saturations = [r['vitals'].get('oxygenSaturation') for r in results if r.get('vitals', {}).get('oxygenSaturation')]
        respiratory_rates = [r['vitals'].get('respiratoryRate') for r in results if r.get('vitals', {}).get('respiratoryRate')]
        temperatures = [r['vitals'].get('temperature') for r in results if r.get('vitals', {}).get('temperature')]
        
        # Calculate BP from available data
        avg_hr = int(np.median(heart_rates)) if heart_rates else 72
        avg_stress = int(np.median(stress_levels)) if stress_levels else 50
        # Create a dummy signal for BP estimation
        dummy_signal = np.array([1.0] * 100)
        bp = self._estimate_blood_pressure(avg_hr, avg_stress, dummy_signal)
        
        return {
            'faceDetected': True,
            'validFrames': len(results),
            'totalFrames': len(frames),
            'frameCount': len(frames),
            'vitals': {
                'heartRate': int(np.median(heart_rates)) if heart_rates else None,
                'stressLevel': int(np.median(stress_levels)) if stress_levels else None,
                'oxygenSaturation': int(np.median(oxygen_saturations)) if oxygen_saturations else None,
                'respiratoryRate': int(np.median(respiratory_rates)) if respiratory_rates else None,
                'temperature': float(np.median(temperatures)) if temperatures else None,
                'bloodPressure': bp,
                'confidence': 0.7,
                'timestamp': time.time() * 1000
            },
            'confidence': '0.70',
            'avgQualityScore': f"{quality_score / face_count:.1f}" if face_count > 0 else '0.0'
        }
    
    def analyze_image_frame(self, image_bytes: bytes) -> Dict:
        """
        Analyze a single image frame for vital signs
        
        Args:
            image_bytes: Image as bytes
            
        Returns:
            Dict with analysis results
        """
        start_time = time.time()
        print("[VitalSigns] Starting single frame analysis...")
        
        # Validate frame quality
        quality_check = self.face_detection_service.validate_frame_quality(image_bytes)
        
        if not quality_check['isValid'] and quality_check['score'] == 0:
            print(f"[VitalSigns] Frame quality check failed: score={quality_check['score']}")
            return {
                'faceDetected': False,
                'vitals': {},
                'qualityScore': 0
            }
        
        print(f"[VitalSigns] Frame quality check passed: score={quality_check['score']}")
        
        # Detect face
        face_result = self.face_detection_service.detect_face(image_bytes)
        
        if not face_result.get('detected'):
            return {
                'faceDetected': False,
                'vitals': {},
                'qualityScore': quality_check['score']
            }
        
        print(f"[VitalSigns] Face detected: confidence={face_result['confidence']:.2f}")
        
        # Extract ROI
        roi_bytes = None
        use_full_image = False
        
        try:
            roi_bytes = self.face_detection_service.extract_roi(
                image_bytes,
                face_result['boundingBox']
            )
            print("[VitalSigns] ROI extracted successfully")
        except Exception as e:
            print(f"[VitalSigns] ROI extraction failed, using full image: {str(e)}")
            roi_bytes = image_bytes
            use_full_image = True
        
        # Extract vital signs
        vitals = self._extract_vital_signs(roi_bytes, use_full_image)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            face_result['confidence'],
            quality_check['score'],
            vitals.get('heartRate'),
            vitals.get('stressLevel'),
            vitals.get('oxygenSaturation'),
            vitals.get('respiratoryRate'),
            vitals.get('temperature')
        )
        
        vitals['confidence'] = confidence
        
        duration = (time.time() - start_time) * 1000
        
        result = {
            'faceDetected': True,
            'vitals': vitals,
            'confidence': f"{confidence:.2f}",
            'qualityScore': quality_check['score'],
            'duration': f"{duration:.0f}ms"
        }
        
        print(f"[VitalSigns] ✓ Frame analysis complete: {result}")
        return result
    
    def _extract_vital_signs(self, roi_bytes: bytes, use_full_image: bool = False) -> Dict:
        """
        Extract vital signs from ROI using PPG and color analysis
        """
        try:
            # Decode image bytes (with fallback support)
            image = decode_image_bytes(roi_bytes)
            
            if image is None:
                return self._default_vitals()
            
            height, width = image.shape[:2]
            
            # Extract red channel (most sensitive to blood flow changes)
            red_channel = image[:, :, 2].flatten().astype(np.float32)
            
            # Calculate heart rate using PPG
            heart_rate = self._calculate_heart_rate_ppg(red_channel)
            
            # Calculate stress level from color analysis
            stress_level = self._calculate_stress_level(image)
            
            # Calculate oxygen saturation
            oxygen_saturation = self._calculate_oxygen_saturation(image)
            
            # Estimate respiratory rate
            respiratory_rate = self._estimate_respiratory_rate(red_channel)

            # Estimate temperature
            temperature = self._estimate_temperature(image)
            
            # Estimate blood pressure based on heart rate, stress level, and signal characteristics
            blood_pressure = self._estimate_blood_pressure(heart_rate, stress_level, red_channel)
            
            return {
                'heartRate': heart_rate,
                'stressLevel': stress_level,
                'oxygenSaturation': oxygen_saturation,
                'respiratoryRate': respiratory_rate,
                'temperature': temperature,
                'bloodPressure': blood_pressure
            }
            
        except Exception as e:
            print(f"[VitalSigns] Error extracting vital signs: {str(e)}")
            return self._default_vitals()
    
    def _calculate_heart_rate_ppg(self, red_channel: np.ndarray) -> int:
        """
        Calculate heart rate using PPG (Photoplethysmography)
        Improved algorithm with autocorrelation
        """
        if len(red_channel) < 100:
            return 72  # Default fallback
        
        # Normalize signal
        mean = np.mean(red_channel)
        normalized = red_channel - mean
        
        # Apply median filter to reduce noise
        normalized = median_filter(normalized, size=5)
        
        # Autocorrelation to find periodic patterns
        max_lag = min(50, len(normalized) // 2)
        autocorr = []
        
        for lag in range(max_lag):
            if lag == 0:
                autocorr.append(np.sum(normalized ** 2) / len(normalized))
            else:
                corr = np.sum(normalized[:-lag] * normalized[lag:]) / (len(normalized) - lag)
                autocorr.append(corr)
        
        autocorr = np.array(autocorr)
        
        # Find peaks in autocorrelation
        peaks, _ = signal.find_peaks(autocorr[1:], height=0)
        peaks = peaks + 1  # Adjust for offset
        
        if len(peaks) >= 2:
            # Calculate intervals between peaks
            intervals = np.diff(peaks)
            avg_interval = np.mean(intervals)
            
            # Assuming ~10 FPS, convert to heart rate
            heart_rate = int(60 / (avg_interval * 0.1))
            
            # Validate heart rate range
            if 50 <= heart_rate <= 120:
                return heart_rate
        
        # Fallback: variance-based estimation
        variance = np.var(normalized)
        std_dev = np.sqrt(variance)
        base_rate = 72
        signal_strength = min(1.0, std_dev / 10.0)
        estimated_hr = base_rate + (signal_strength - 0.5) * 20
        
        return int(np.clip(estimated_hr, 60, 100))
    
    def _calculate_stress_level(self, image: np.ndarray) -> int:
        """
        Calculate stress level from color variance and skin tone analysis
        """
        # Convert to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Calculate color variance
        r_var = np.var(rgb[:, :, 0])
        g_var = np.var(rgb[:, :, 1])
        b_var = np.var(rgb[:, :, 2])
        total_variance = (r_var + g_var + b_var) / 3
        
        # Calculate R/G ratio (stress affects skin tone)
        r_mean = np.mean(rgb[:, :, 0])
        g_mean = np.mean(rgb[:, :, 1])
        rg_ratio = r_mean / (g_mean + 1e-6)  # Avoid division by zero
        
        # Normalize variance (typical range: 0-500)
        normalized_variance = min(1.0, total_variance / 500.0)
        
        # Stress estimation: higher variance and color variation = higher stress
        stress_base = 30
        stress_variance_component = normalized_variance * 30
        stress_color_component = abs(rg_ratio - 1.2) * 20  # Optimal R/G is around 1.2
        
        stress_level = stress_base + stress_variance_component + stress_color_component
        
        return int(np.clip(stress_level, 0, 100))
    
    def _calculate_oxygen_saturation(self, image: np.ndarray) -> int:
        """
        Calculate oxygen saturation (SpO2) using R/B ratio
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Calculate R/B ratios for all pixels
        r_channel = rgb[:, :, 0].astype(np.float32)
        b_channel = rgb[:, :, 1].astype(np.float32)
        
        # Avoid division by zero
        b_channel[b_channel == 0] = 1
        ratios = r_channel / b_channel
        
        # Use median for robustness
        median_ratio = np.median(ratios)
        
        # Optimal R/B ratio for healthy SpO2 is around 1.0-1.2
        # Deviation from this indicates lower SpO2
        optimal_ratio = 1.1
        deviation = abs(median_ratio - optimal_ratio)
        
        # Estimate SpO2 (95-100% is normal)
        spo2 = 100 - (deviation * 10)
        
        return int(np.clip(spo2, 95, 100))
    
    def _estimate_respiratory_rate(self, red_channel: np.ndarray) -> int:
        """
        Estimate respiratory rate from signal variation
        Uses low-frequency signal analysis
        """
        if len(red_channel) < 50:
            return 16  # Default fallback
        
        # Normalize signal
        mean = np.mean(red_channel)
        normalized = red_channel - mean
        
        # Apply moving average to extract low-frequency components
        window_size = min(10, len(normalized) // 5)
        if window_size < 3:
            window_size = 3
        
        smoothed = np.convolve(normalized, np.ones(window_size)/window_size, mode='same')
        
        # Count zero crossings to estimate breathing cycles
        zero_crossings = np.where(np.diff(np.sign(smoothed)))[0]
        
        if len(zero_crossings) >= 2:
            # Calculate average interval between zero crossings
            intervals = np.diff(zero_crossings)
            avg_interval = np.mean(intervals)
            
            # Convert to breaths per minute (assuming ~10 FPS)
            breaths_per_min = int(60 / (avg_interval * 0.1 * 2))  # *2 because one cycle = 2 zero crossings
            
            # Validate range
            if 10 <= breaths_per_min <= 25:
                return breaths_per_min
        
        # Fallback: variance-based estimation
        variance = np.var(normalized)
        std_dev = np.sqrt(variance)
        base_rate = 16
        signal_strength = min(1.0, std_dev / 5.0)
        estimated_rr = base_rate + (signal_strength - 0.5) * 4
        
        return int(np.clip(estimated_rr, 10, 25))
    
    def _calculate_confidence(
        self,
        face_confidence: float,
        quality_score: float,
        heart_rate: Optional[int] = None,
        stress_level: Optional[int] = None,
        oxygen_saturation: Optional[int] = None,
        respiratory_rate: Optional[int] = None,
        temperature: Optional[float] = None,
        signal_quality: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Enhanced confidence score calculation with signal quality metrics
        Includes: face detection, image quality, vital completeness, reasonableness,
                  signal-to-noise ratio, motion level, and signal stability
        """
        # Base confidence from face detection
        confidence = face_confidence * 0.35  # Slightly reduced to make room for new metrics
        
        # Quality score contribution (image quality)
        confidence += (quality_score / 100.0) * 0.25
        
        # Vital signs completeness
        vitals_count = sum([
            heart_rate is not None,
            stress_level is not None,
            oxygen_saturation is not None,
            respiratory_rate is not None,
            temperature is not None
        ])
        completeness = vitals_count / 5.0
        confidence += completeness * 0.15
        
        # Enhanced reasonableness check (more sophisticated)
        reasonableness = 1.0
        issues = 0
        
        if heart_rate and not (50 <= heart_rate <= 120):
            issues += 1
            if heart_rate < 40 or heart_rate > 150:  # Very abnormal
                reasonableness *= 0.6
            else:
                reasonableness *= 0.85
        
        if stress_level and not (0 <= stress_level <= 100):
            issues += 1
            reasonableness *= 0.85
        
        if oxygen_saturation and not (95 <= oxygen_saturation <= 100):
            issues += 1
            if oxygen_saturation < 90:  # Very abnormal
                reasonableness *= 0.6
            else:
                reasonableness *= 0.85
        
        if respiratory_rate and not (10 <= respiratory_rate <= 25):
            issues += 1
            if respiratory_rate < 8 or respiratory_rate > 30:  # Very abnormal
                reasonableness *= 0.6
            else:
                reasonableness *= 0.85
        
        if temperature and not (35.5 <= temperature <= 39.0):
            issues += 1
            if temperature < 34 or temperature > 40:  # Very abnormal
                reasonableness *= 0.6
            else:
                reasonableness *= 0.85
        
        # Multiple issues reduce confidence more
        if issues > 2:
            reasonableness *= 0.8  # Additional penalty for multiple issues
        
        confidence += reasonableness * 0.10
        
        # Signal quality metrics (if available)
        if signal_quality:
            # Signal-to-noise ratio contribution (0-10% of confidence)
            snr = signal_quality.get('snr', 0)
            if snr > 0:
                # Normalize SNR (typical range: 5-30 dB, target: >10 dB)
                snr_norm = min(1.0, max(0.0, (snr - 5) / 25.0))
                confidence += snr_norm * 0.10
            
            # Signal stability contribution (0-5% of confidence)
            stability = signal_quality.get('stability', 0)
            if stability > 0:
                # Higher stability = higher confidence
                confidence += stability * 0.05
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _analyze_with_overlapping_windows(
        self,
        rois: List[bytes],
        sensor_data: Optional[Dict] = None,
        face_detected_count: int = 0,
        total_quality_score: float = 0,
        total_frame_count: int = 0,
        user_profile: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze short video using overlapping sliding windows for better accuracy
        Creates multiple overlapping windows from the same frames and averages results
        """
        WINDOW_SIZE = 7  # Size of each window
        OVERLAP = 3  # Overlap between windows (frames)
        STEP_SIZE = WINDOW_SIZE - OVERLAP  # Step between windows
        
        if len(rois) < WINDOW_SIZE:
            # Too short even for sliding windows, use regular analysis
            print(f"[VitalSigns] Video too short for sliding windows ({len(rois)} frames), using regular analysis")
            # Fallback to regular analysis with what we have
            signals = self._extract_multi_roi_signals(rois)
            if len(signals['red']) < 5:
                return self._fallback_analysis_safe(face_detected_count, total_quality_score, total_frame_count)
            # Continue with regular processing
            return self._analyze_single_window(rois, signals, sensor_data, face_detected_count, total_quality_score, total_frame_count)
        
        # Create overlapping windows
        windows = []
        start_idx = 0
        while start_idx + WINDOW_SIZE <= len(rois):
            window = rois[start_idx:start_idx + WINDOW_SIZE]
            windows.append(window)
            start_idx += STEP_SIZE
        
        # If we didn't cover all frames, add a final window from the end
        if start_idx < len(rois):
            final_window = rois[-WINDOW_SIZE:]
            if final_window != windows[-1] if windows else True:  # Don't duplicate last window
                windows.append(final_window)
        
        print(f"[VitalSigns] Created {len(windows)} overlapping windows from {len(rois)} frames")
        
        # Analyze each window
        window_results = []
        for i, window in enumerate(windows):
            try:
                # Extract signals from window
                signals = self._extract_multi_roi_signals(window)
                
                if len(signals['red']) < 5:
                    print(f"[VitalSigns] Window {i+1} has insufficient signals, skipping")
                    continue
                
                # Analyze this window with user profile for calibration
                result = self._analyze_single_window(window, signals, sensor_data, face_detected_count, total_quality_score, total_frame_count, user_profile=user_profile)
                
                # Only include valid results
                if result and result.get('vitals'):
                    vitals = result['vitals']
                    # Check if vital signs are valid (not None, not 0)
                    if (vitals.get('heartRate') and vitals.get('heartRate') > 0 and
                        vitals.get('confidence') and float(vitals.get('confidence', 0)) > 0.5):
                        window_results.append(result)
                        
            except Exception as e:
                print(f"[VitalSigns] Error analyzing window {i+1}: {str(e)}")
                continue
        
        if not window_results:
            print(f"[VitalSigns] No valid window results, using fallback")
            return self._fallback_analysis_safe(face_detected_count, total_quality_score, total_frame_count)
        
        # Average results from all windows
        print(f"[VitalSigns] Averaging results from {len(window_results)} valid windows")
        return self._average_window_results(window_results, face_detected_count, total_quality_score, total_frame_count)
    
    def _analyze_single_window(
        self,
        rois: List[bytes],
        signals: Dict[str, np.ndarray],
        sensor_data: Optional[Dict] = None,
        face_detected_count: int = 0,
        total_quality_score: float = 0,
        total_frame_count: int = 0,
        user_profile: Optional[Dict] = None
    ) -> Dict:
        """Analyze a single window of frames"""
        # Estimate FPS (assume ~10 FPS for short videos)
        estimated_fps = 10.0
        if len(signals['red']) > 1:
            # Estimate FPS from signal length (rough estimate)
            estimated_fps = max(5.0, min(30.0, len(signals['red']) * 2))  # Rough estimate
        
        # Preprocess signals
        red_signal = self._preprocess_signal(signals['red'], sensor_data, fps=estimated_fps)
        green_signal = self._preprocess_signal(signals['green'], sensor_data, fps=estimated_fps)
        blue_signal = self._preprocess_signal(signals['blue'], sensor_data, fps=estimated_fps)
        
        # Calculate vital signs
        heart_rate = self._calculate_heart_rate_ensemble(red_signal, green_signal, blue_signal, fps=estimated_fps)
        respiratory_rate = self._calculate_respiratory_rate_ensemble(red_signal, fps=estimated_fps)
        
        # For other vitals, use available frames
        stress_level = self._calculate_stress_from_rois(rois[-min(10, len(rois)):])
        oxygen_saturation = self._calculate_oxygen_saturation_improved(rois[-min(10, len(rois)):], red_signal, green_signal, blue_signal, user_profile)
        temperature = self._estimate_temperature_from_rois(rois[-min(10, len(rois)):], sensor_data, user_profile)
        blood_pressure = self._estimate_blood_pressure(heart_rate, stress_level, red_signal)
        
        # Calculate signal quality metrics
        signal_quality_metrics = {}
        try:
            if len(red_signal) >= 5:
                signal_quality_metrics = self._calculate_signal_quality_metrics(
                    red_signal, green_signal, blue_signal, motion_level=None
                )
        except Exception:
            signal_quality_metrics = {}
        
        # Calculate confidence
        avg_quality_score = total_quality_score / face_detected_count if face_detected_count > 0 else 0
        confidence = self._calculate_confidence(
            face_detected_count / total_frame_count if total_frame_count > 0 else 0,
            avg_quality_score,
            heart_rate,
            stress_level,
            oxygen_saturation,
            respiratory_rate,
            temperature,
            signal_quality=signal_quality_metrics if signal_quality_metrics else None
        )
        
        return {
            'vitals': {
                'heartRate': heart_rate,
                'stressLevel': stress_level,
                'oxygenSaturation': oxygen_saturation,
                'respiratoryRate': respiratory_rate,
                'temperature': temperature,
                'bloodPressure': blood_pressure,
                'confidence': f"{confidence:.2f}",
                'timestamp': time.time() * 1000
            },
            'confidence': f"{confidence:.2f}",
            'avgQualityScore': f"{avg_quality_score:.1f}"
        }
    
    def _average_window_results(
        self,
        window_results: List[Dict],
        face_detected_count: int,
        total_quality_score: float,
        total_frame_count: int
    ) -> Dict:
        """Average results from multiple overlapping windows"""
        heart_rates = []
        stress_levels = []
        oxygen_saturations = []
        respiratory_rates = []
        temperatures = []
        systolic_bps = []
        diastolic_bps = []
        confidences = []
        
        for result in window_results:
            vitals = result.get('vitals', {})
            if vitals.get('heartRate') and vitals.get('heartRate') > 0:
                heart_rates.append(vitals['heartRate'])
            if vitals.get('stressLevel') is not None:
                stress_levels.append(vitals['stressLevel'])
            if vitals.get('oxygenSaturation') and vitals.get('oxygenSaturation') > 0:
                oxygen_saturations.append(vitals['oxygenSaturation'])
            if vitals.get('respiratoryRate') and vitals.get('respiratoryRate') > 0:
                respiratory_rates.append(vitals['respiratoryRate'])
            if vitals.get('temperature') and vitals.get('temperature') > 0:
                temperatures.append(vitals['temperature'])
            if vitals.get('bloodPressure'):
                bp = vitals['bloodPressure']
                if bp.get('systolic'):
                    systolic_bps.append(bp['systolic'])
                if bp.get('diastolic'):
                    diastolic_bps.append(bp['diastolic'])
            if vitals.get('confidence'):
                try:
                    confidences.append(float(vitals['confidence']))
                except (ValueError, TypeError):
                    pass
        
        # Use median for robustness (less sensitive to outliers)
        avg_heart_rate = int(np.median(heart_rates)) if heart_rates else None
        avg_stress = int(np.median(stress_levels)) if stress_levels else None
        avg_spo2 = int(np.median(oxygen_saturations)) if oxygen_saturations else None
        avg_rr = int(np.median(respiratory_rates)) if respiratory_rates else None
        avg_temp = float(np.median(temperatures)) if temperatures else None
        avg_systolic = int(np.median(systolic_bps)) if systolic_bps else None
        avg_diastolic = int(np.median(diastolic_bps)) if diastolic_bps else None
        avg_confidence = float(np.median(confidences)) if confidences else 0.7
        
        return {
            'vitals': {
                'heartRate': avg_heart_rate,
                'stressLevel': avg_stress,
                'oxygenSaturation': avg_spo2,
                'respiratoryRate': avg_rr,
                'temperature': avg_temp,
                'bloodPressure': {
                    'systolic': avg_systolic,
                    'diastolic': avg_diastolic
                } if avg_systolic and avg_diastolic else None,
                'confidence': f"{avg_confidence:.2f}",
                'timestamp': time.time() * 1000
            },
            'confidence': f"{avg_confidence:.2f}",
            'avgQualityScore': f"{total_quality_score / face_detected_count:.1f}" if face_detected_count > 0 else '0.0',
            'windowsAnalyzed': len(window_results)
        }
    
    def _calculate_signal_quality_metrics(
        self,
        red_signal: np.ndarray,
        green_signal: np.ndarray,
        blue_signal: np.ndarray,
        motion_level: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate signal quality metrics: SNR, stability, motion level
        """
        metrics = {}
        
        if len(red_signal) >= 10:
            # Calculate signal-to-noise ratio (SNR)
            # SNR = signal_power / noise_power
            signal_power = np.var(red_signal)
            
            # Estimate noise as high-frequency component (using derivative)
            if len(red_signal) > 1:
                noise_estimate = np.std(np.diff(red_signal))
                noise_power = noise_estimate ** 2 if noise_estimate > 0 else 1e-6
                
                # SNR in dB
                snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
                metrics['snr'] = float(max(0, snr_db))
            
            # Signal stability (inverse of coefficient of variation)
            signal_mean = np.mean(red_signal)
            signal_std = np.std(red_signal)
            if signal_mean > 0:
                cv = signal_std / signal_mean  # Coefficient of variation
                stability = 1.0 / (1.0 + cv)  # Normalize to 0-1
                metrics['stability'] = float(np.clip(stability, 0.0, 1.0))
        
        # Motion level (if available)
        if motion_level is not None:
            # Lower motion = higher quality (inverse relationship)
            # Motion level normalized (0-10 scale), convert to quality (0-1)
            motion_quality = max(0.0, 1.0 - motion_level / 10.0)
            metrics['motion_quality'] = float(motion_quality)
        
        return metrics
    
    def _calculate_heart_rate_temporal(self, red_signal: np.ndarray, fps: float = 5.0) -> int:
        """
        Calculate heart rate from temporal signal using FFT
        Improved for low frame rates with better frequency resolution
        """
        if len(red_signal) < 10:
            print(f"[VitalSigns] FFT: Signal too short ({len(red_signal)} samples)")
            return 72  # Default fallback
        
        # Check signal variation
        signal_std = np.std(red_signal)
        signal_range = np.max(red_signal) - np.min(red_signal)
        
        if signal_std < 0.1 or signal_range < 0.5:
            print(f"[VitalSigns] FFT: Signal too flat (std={signal_std:.3f}, range={signal_range:.3f})")
            return 72
        
        # Normalize signal
        mean = np.mean(red_signal)
        normalized = red_signal - mean
        
        # Apply bandpass filter (0.8-3.5 Hz for heart rate: 48-210 BPM, tighter range for accuracy)
        try:
            nyquist = fps / 2.0
            # Ensure filter bounds are within Nyquist frequency
            if nyquist < 0.8:
                print(f"[VitalSigns] FFT: FPS too low ({fps:.2f} Hz), Nyquist={nyquist:.2f} Hz")
                # Use a simpler approach for very low FPS
                return self._calculate_heart_rate_autocorrelation(normalized, fps)
            
            low = max(0.8 / nyquist, 0.1)  # ~48 BPM minimum
            high = min(3.5 / nyquist, 0.95)  # ~210 BPM maximum, stay below Nyquist
            
            if low >= high:
                low = 0.1
                high = 0.9
            
            b, a = butter(4, [low, high], btype='band')
            filtered = filtfilt(b, a, normalized)
        except Exception as e:
            print(f"[VitalSigns] FFT: Filter failed: {str(e)}")
            filtered = normalized
        
        # Apply windowing to reduce spectral leakage (use Blackman window for better frequency resolution)
        windowed = filtered * np.blackman(len(filtered))
        
        # FFT with zero-padding for better frequency resolution
        # Pad to next power of 2 for faster FFT
        fft_size = 2 ** int(np.ceil(np.log2(len(windowed) * 4)))  # 4x padding for better resolution
        fft_result = np.fft.rfft(windowed, n=fft_size)
        freqs = np.fft.rfftfreq(fft_size, 1.0/fps)
        magnitude = np.abs(fft_result)
        
        # Find peak in heart rate range (0.8-3.5 Hz = 48-210 BPM)
        hr_range = (freqs >= 0.8) & (freqs <= 3.5)
        if not np.any(hr_range):
            print(f"[VitalSigns] FFT: No frequencies in HR range")
            return 72
        
        hr_freqs = freqs[hr_range]
        hr_magnitude = magnitude[hr_range]
        
        # Find dominant frequency (weighted average of top peaks for robustness)
        top_n = min(3, len(hr_magnitude))
        top_indices = np.argsort(hr_magnitude)[-top_n:][::-1]
        top_magnitudes = hr_magnitude[top_indices]
        top_freqs = hr_freqs[top_indices]
        
        # Weighted average (more weight to stronger peaks)
        if np.sum(top_magnitudes) > 0:
            weights = top_magnitudes / np.sum(top_magnitudes)
            peak_freq = np.average(top_freqs, weights=weights)
            peak_magnitude = np.max(top_magnitudes)
        else:
            peak_idx = np.argmax(hr_magnitude)
            peak_freq = hr_freqs[peak_idx]
            peak_magnitude = hr_magnitude[peak_idx]
        
        # Check if peak is significant (at least 15% of max magnitude for better accuracy)
        max_magnitude = np.max(magnitude)
        if peak_magnitude < max_magnitude * 0.15:
            print(f"[VitalSigns] FFT: Peak too weak ({peak_magnitude:.2f} vs {max_magnitude:.2f})")
            return 72
        
        # Convert to BPM
        heart_rate = int(round(peak_freq * 60))
        
        print(f"[VitalSigns] FFT: Found peak at {peak_freq:.3f} Hz = {heart_rate} BPM (magnitude: {peak_magnitude:.2f})")
        
        # Validate range (50-120 BPM for normal resting heart rate)
        if 50 <= heart_rate <= 120:
            return heart_rate
        
        # Try autocorrelation as fallback if FFT result is out of range
        print(f"[VitalSigns] FFT: Result {heart_rate} BPM out of range, trying autocorrelation")
        hr_autocorr = self._calculate_heart_rate_autocorrelation(normalized, fps)
        if 50 <= hr_autocorr <= 120:
            return hr_autocorr
        
        # Final fallback: clip to reasonable range
        print(f"[VitalSigns] FFT: All methods out of range, clipping to 60-100")
        return int(np.clip(heart_rate, 60, 100))
    
    def _calculate_respiratory_rate_temporal(self, red_signal: np.ndarray, fps: float = 10.0) -> int:
        """
        Calculate respiratory rate from temporal signal using FFT
        Respiratory rate is in 0.15-0.5 Hz range (9-30 breaths/min)
        """
        if len(red_signal) < 20:
            return 16  # Default fallback
        
        # Normalize signal
        mean = np.mean(red_signal)
        normalized = red_signal - mean
        
        # Apply low-pass filter for respiratory rate (0.15-0.5 Hz)
        try:
            nyquist = fps / 2.0
            high = 0.5 / nyquist  # ~30 breaths/min
            b, a = butter(4, high, btype='low')
            filtered = filtfilt(b, a, normalized)
        except:
            filtered = normalized
        
        # Apply windowing
        windowed = filtered * np.hanning(len(filtered))
        
        # FFT
        fft = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(len(windowed), 1.0/fps)
        magnitude = np.abs(fft)
        
        # Find peak in respiratory range (0.15-0.5 Hz = 9-30 breaths/min)
        rr_range = (freqs >= 0.15) & (freqs <= 0.5)
        if not np.any(rr_range):
            return 16
        
        rr_freqs = freqs[rr_range]
        rr_magnitude = magnitude[rr_range]
        
        # Find dominant frequency
        peak_idx = np.argmax(rr_magnitude)
        peak_freq = rr_freqs[peak_idx]
        
        # Convert to breaths per minute
        respiratory_rate = int(peak_freq * 60)
        
        # Validate range
        if 10 <= respiratory_rate <= 25:
            return respiratory_rate
        
        # Fallback
        return int(np.clip(respiratory_rate, 12, 20))
    
    def _calculate_stress_from_rois(self, rois: List[bytes]) -> int:
        """Calculate stress level from multiple ROIs"""
        stress_levels = []
        for roi_bytes in rois:
            try:
                image = decode_image_bytes(roi_bytes)
                if image is not None:
                    stress = self._calculate_stress_level(image)
                    stress_levels.append(stress)
            except:
                continue
        
        return int(np.median(stress_levels)) if stress_levels else 30
    
    def _calculate_oxygen_saturation_from_rois(self, rois: List[bytes]) -> int:
        """Calculate oxygen saturation from multiple ROIs"""
        spo2_levels = []
        for roi_bytes in rois:
            try:
                image = decode_image_bytes(roi_bytes)
                if image is not None:
                    spo2 = self._calculate_oxygen_saturation(image)
                    spo2_levels.append(spo2)
            except:
                continue
        
        return int(np.median(spo2_levels)) if spo2_levels else 98
    
    def _adjust_quality_from_sensors(self, sensor_data: Dict) -> float:
        """
        Adjust quality score based on sensor data
        Returns adjustment value (negative = reduce quality, positive = increase)
        """
        adjustment = 0.0
        
        # Check accelerometer (motion)
        if 'accelerometer' in sensor_data and sensor_data['accelerometer']:
            accel = sensor_data['accelerometer']
            magnitude = accel.get('magnitude', 0)
            if magnitude > 2.0:  # Too much movement
                adjustment -= 15
            elif magnitude > 1.5:
                adjustment -= 10
        
        # Check gyroscope (rotation)
        if 'gyroscope' in sensor_data and sensor_data['gyroscope']:
            gyro = sensor_data['gyroscope']
            magnitude = gyro.get('magnitude', 0)
            if magnitude > 0.5:  # Too much rotation
                adjustment -= 10
        
        # Check proximity
        if 'proximity' in sensor_data and sensor_data['proximity']:
            prox = sensor_data['proximity']
            distance = prox.get('distance', 20)
            if distance < 5 or distance > 30:  # Too close or too far
                adjustment -= 10
        
        # Check ambient light
        if 'ambientLight' in sensor_data and sensor_data['ambientLight']:
            light = sensor_data['ambientLight']
            illuminance = light.get('illuminance', 200)
            if illuminance < 50:  # Too dim
                adjustment -= 15
            elif illuminance > 500:  # Too bright
                adjustment -= 5
        
        return adjustment
    
    def _detect_skin_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Detect skin pixels using color-based skin detection
        Uses YCrCb color space for better skin detection across different skin tones
        Returns binary mask where 1 = skin pixel, 0 = non-skin pixel
        """
        try:
            # Convert to YCrCb color space (better for skin detection)
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            
            # Skin color ranges in YCrCb space (works for various skin tones)
            # These ranges are based on research and work for most skin tones
            lower_skin = np.array([0, 135, 85], dtype=np.uint8)
            upper_skin = np.array([255, 180, 135], dtype=np.uint8)
            
            # Create mask for skin pixels
            skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
            
            # Additional HSV-based detection for better coverage
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Hue range for skin (0-20 and 160-180 for different lighting)
            lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
            upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
            
            # Combine both masks (OR operation)
            combined_mask = cv2.bitwise_or(skin_mask, skin_mask_hsv)
            
            # Morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # If mask is too small (< 5% of image), fall back to forehead region
            mask_ratio = np.sum(combined_mask > 0) / (image.shape[0] * image.shape[1])
            if mask_ratio < 0.05:
                # Fallback: use top 30% of image (forehead region)
                h, w = image.shape[:2]
                combined_mask = np.zeros((h, w), dtype=np.uint8)
                combined_mask[:int(h * 0.3), :] = 255
            
            return combined_mask
            
        except Exception as e:
            print(f"[VitalSigns] Skin mask detection failed: {str(e)}")
            # Fallback: return mask for top 30% (forehead)
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[:int(h * 0.3), :] = 255
            return mask
    
    def _extract_multi_roi_signals(self, rois: List[bytes], use_skin_mask: bool = True) -> Dict[str, np.ndarray]:
        """
        Extract signals from multiple regions of interest with optional skin mask detection
        Improved PPG signal extraction with focus on skin pixels only
        """
        red_signals = []
        green_signals = []
        blue_signals = []
        
        for roi_bytes in rois:
            try:
                image = decode_image_bytes(roi_bytes)
                if image is None:
                    continue
                
                h, w = image.shape[:2]
                
                if use_skin_mask:
                    # Use skin mask detection for more accurate signal extraction
                    skin_mask = self._detect_skin_mask(image)
                    
                    # Extract signals only from skin pixels
                    # Apply mask to each channel
                    red_channel = image[:, :, 2]  # BGR format
                    green_channel = image[:, :, 1]
                    blue_channel = image[:, :, 0]
                    
                    # Mask skin pixels
                    red_skin = red_channel[skin_mask > 0]
                    green_skin = green_channel[skin_mask > 0]
                    blue_skin = blue_channel[skin_mask > 0]
                    
                    if len(red_skin) > 0:
                        red_combined = float(np.mean(red_skin))
                        green_combined = float(np.mean(green_skin))
                        blue_combined = float(np.mean(blue_skin))
                    else:
                        # Fallback if no skin pixels detected
                        red_combined = float(np.mean(red_channel))
                        green_combined = float(np.mean(green_channel))
                        blue_combined = float(np.mean(blue_channel))
                else:
                    # Original method: fixed regions with weights
                    # Forehead region (top 25%) - best for PPG (highest weight)
                    forehead_top = int(h * 0.05)  # Top 5% - most reliable
                    forehead_bottom = int(h * 0.3)  # Top 30%
                    forehead = image[forehead_top:forehead_bottom, :]
                    
                    # Middle forehead (10-20%) - secondary PPG region
                    mid_forehead = image[int(h*0.1):int(h*0.2), :]
                    
                    # Cheek regions (middle 30-50%) - for color analysis
                    cheeks = image[int(h*0.3):int(h*0.5), :]
                    
                    # Weighted combination: 70% forehead (50% top + 20% mid), 30% cheeks
                    red_top_forehead = np.mean(forehead[:, :, 2]) if forehead.size > 0 else 0
                    red_mid_forehead = np.mean(mid_forehead[:, :, 2]) if mid_forehead.size > 0 else 0
                    red_cheeks = np.mean(cheeks[:, :, 2]) if cheeks.size > 0 else 0
                    red_combined = 0.5 * red_top_forehead + 0.2 * red_mid_forehead + 0.3 * red_cheeks
                    
                    green_top_forehead = np.mean(forehead[:, :, 1]) if forehead.size > 0 else 0
                    green_mid_forehead = np.mean(mid_forehead[:, :, 1]) if mid_forehead.size > 0 else 0
                    green_cheeks = np.mean(cheeks[:, :, 1]) if cheeks.size > 0 else 0
                    green_combined = 0.5 * green_top_forehead + 0.2 * green_mid_forehead + 0.3 * green_cheeks
                    
                    blue_top_forehead = np.mean(forehead[:, :, 0]) if forehead.size > 0 else 0
                    blue_mid_forehead = np.mean(mid_forehead[:, :, 0]) if mid_forehead.size > 0 else 0
                    blue_cheeks = np.mean(cheeks[:, :, 0]) if cheeks.size > 0 else 0
                    blue_combined = 0.5 * blue_top_forehead + 0.2 * blue_mid_forehead + 0.3 * blue_cheeks
                
                red_signals.append(red_combined)
                green_signals.append(green_combined)
                blue_signals.append(blue_combined)
                
            except Exception as e:
                print(f"[VitalSigns] Error extracting multi-ROI signal: {str(e)}")
                continue
        
        return {
            'red': np.array(red_signals) if red_signals else np.array([]),
            'green': np.array(green_signals) if green_signals else np.array([]),
            'blue': np.array(blue_signals) if blue_signals else np.array([])
        }
    
    def _validate_signal_quality(self, signals: Dict[str, np.ndarray]) -> Dict:
        """
        Validate that signals have enough variation to detect vital signs
        """
        red = signals['red']
        green = signals['green']
        blue = signals['blue']
        
        # Check if signals have enough variation
        red_std = np.std(red)
        green_std = np.std(green)
        blue_std = np.std(blue)
        
        # Minimum standard deviation threshold (signals need to vary)
        min_std = 0.5  # Minimum variation needed
        
        if red_std < min_std and green_std < min_std and blue_std < min_std:
            return {
                'isValid': False,
                'reason': f'Insufficient signal variation (std: R={red_std:.2f}, G={green_std:.2f}, B={blue_std:.2f})'
            }
        
        # Check signal-to-noise ratio (rough estimate)
        red_range = np.max(red) - np.min(red)
        if red_range < 1.0:  # Need at least 1 unit of variation
            return {
                'isValid': False,
                'reason': f'Signal too flat (range: {red_range:.2f})'
            }
        
        return {'isValid': True, 'reason': 'OK'}
    
    def _preprocess_signal(self, signal: np.ndarray, sensor_data: Optional[Dict] = None, fps: float = 10.0) -> np.ndarray:
        """
        Advanced signal preprocessing with adaptive filtering and interpolation
        Preserves signal variation while removing noise, with interpolation for low frame rates
        """
        if len(signal) < 3:
            return signal
        
        original_std = np.std(signal)
        original_range = np.max(signal) - np.min(signal)
        
        # 1. Remove DC component (but keep a copy for reference)
        dc_component = np.mean(signal)
        signal_centered = signal - dc_component
        
        # 2. Interpolate signal for low frame rates to improve frequency resolution
        # If FPS < 5, interpolate to at least 5 Hz equivalent for better FFT analysis
        if fps < 5.0 and len(signal_centered) > 5:
            original_indices = np.arange(len(signal_centered))
            # Target: at least 5 Hz sampling rate
            target_length = int(len(signal_centered) * (5.0 / fps))
            target_length = max(target_length, len(signal_centered))  # Don't downsample
            new_indices = np.linspace(0, len(signal_centered) - 1, target_length)
            interp_func = interp1d(original_indices, signal_centered, kind='cubic', fill_value='extrapolate')
            signal_centered = interp_func(new_indices)
            print(f"[VitalSigns] Interpolated signal from {len(signal)} to {len(signal_centered)} samples (FPS: {fps:.2f} -> 5.0)")
        
        # 3. Outlier detection and removal (using IQR method) - be conservative
        q1, q3 = np.percentile(signal_centered, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            # Use wider bounds to avoid removing valid signal variation
            lower_bound = q1 - 2.5 * iqr  # Increased from 1.5 to 2.5
            upper_bound = q3 + 2.5 * iqr
            outliers = (signal_centered < lower_bound) | (signal_centered > upper_bound)
            if np.any(outliers) and np.sum(outliers) < len(signal_centered) * 0.1:  # Only if <10% are outliers
                # Replace outliers with median
                signal_centered[outliers] = np.median(signal_centered[~outliers])
        
        # 4. Adaptive denoising based on motion - lighter filtering to preserve signal
        if sensor_data and sensor_data.get('accelerometer'):
            motion = sensor_data['accelerometer'].get('magnitude', 0)
            if motion > 1.5:
                # High motion: moderate filtering
                signal_centered = gaussian_filter1d(signal_centered, sigma=1.0)
            else:
                # Low motion: very light filtering
                signal_centered = gaussian_filter1d(signal_centered, sigma=0.5)
        else:
            signal_centered = gaussian_filter1d(signal_centered, sigma=0.7)
        
        # 5. ICA-like filtering using PCA-based component separation (simpler than full ICA)
        # Separate pulse signal from motion artifacts using principal component analysis
        if len(signal_centered) >= 10:
            try:
                # Create signal matrix from multiple channel signals if available
                # For single channel, use temporal segments for component analysis
                # This is a simplified ICA approach that works well for PPG signals
                
                # Create segments for component analysis (overlapping windows)
                segment_size = min(5, len(signal_centered) // 2)
                if segment_size >= 3:
                    # Extract multiple overlapping segments
                    segments = []
                    for j in range(0, len(signal_centered) - segment_size + 1, max(1, segment_size // 2)):
                        segment = signal_centered[j:j+segment_size]
                        segments.append(segment)
                    
                    if len(segments) >= 3:
                        # Stack segments into matrix
                        segments_matrix = np.array(segments)
                        
                        # Simple PCA-based separation (eigenvalue decomposition)
                        # Center the data
                        segments_centered = segments_matrix - np.mean(segments_matrix, axis=0)
                        
                        # Covariance matrix
                        cov = np.cov(segments_centered.T)
                        
                        # Eigenvalue decomposition
                        eigenvals, eigenvecs = np.linalg.eig(cov)
                        
                        # Get principal components (sorted by eigenvalue)
                        sorted_indices = np.argsort(eigenvals)[::-1]
                        eigenvecs_sorted = eigenvecs[:, sorted_indices]
                        
                        # Project signal onto first principal component (most variance = pulse signal)
                        # Use the component with highest frequency content (pulse signal)
                        signal_projected = np.dot(signal_centered, eigenvecs_sorted[:, 0])
                        
                        # Normalize projection
                        if np.std(signal_projected) > 0:
                            signal_centered = (signal_projected - np.mean(signal_projected)) / np.std(signal_projected) * np.std(signal_centered)
                            
            except Exception as ica_error:
                # If ICA/PCA fails, continue with original signal
                print(f"[VitalSigns] ICA-like filtering failed: {str(ica_error)}")
                pass
        
        # 6. Savitzky-Golay filter - only if signal is long enough and we have variation
        if len(signal_centered) >= 7 and np.std(signal_centered) > 0.1:
            try:
                window_length = min(7, len(signal_centered) if len(signal_centered) % 2 == 1 else len(signal_centered) - 1)
                if window_length >= 5:
                    signal_centered = savgol_filter(signal_centered, window_length, 3)
            except:
                pass
        
        # Verify we didn't remove all variation
        final_std = np.std(signal_centered)
        final_range = np.max(signal_centered) - np.min(signal_centered)
        
        # If filtering removed too much variation, use less aggressive preprocessing
        if final_std < original_std * 0.3 or final_range < original_range * 0.3:
            print(f"[VitalSigns] Warning: Preprocessing removed too much variation, using lighter filtering")
            # Revert to lighter preprocessing
            signal_centered = signal - dc_component
            signal_centered = gaussian_filter1d(signal_centered, sigma=0.3)  # Very light filtering only
        
        return signal_centered
    
    def _detect_frame_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> Tuple[bool, float]:
        """
        Detect motion between two frames using optical flow
        Returns: (has_significant_motion, motion_magnitude)
        """
        try:
            # Calculate optical flow using Lucas-Kanade method
            corners_prev = cv2.goodFeaturesToTrack(
                prev_frame,
                maxCorners=100,
                qualityLevel=0.3,
                minDistance=7,
                blockSize=7
            )
            
            if corners_prev is None or len(corners_prev) < 10:
                return False, 0.0
            
            # Calculate optical flow
            corners_next, status, err = cv2.calcOpticalFlowPyrLK(
                prev_frame, curr_frame, corners_prev, None
            )
            
            # Filter valid points
            valid_prev = corners_prev[status == 1]
            valid_next = corners_next[status == 1]
            
            if len(valid_prev) < 10 or len(valid_next) < 10:
                return False, 0.0
            
            # Calculate flow vectors and magnitudes
            flow_vectors = valid_next - valid_prev
            flow_magnitudes = np.sqrt(flow_vectors[:, 0]**2 + flow_vectors[:, 1]**2)
            avg_motion = float(np.mean(flow_magnitudes))
            
            # Normalize by frame size
            frame_height, frame_width = prev_frame.shape[:2]
            normalized_motion = avg_motion / max(frame_width, frame_height) * 100
            
            # Threshold: motion > 2.0 indicates significant movement
            has_significant_motion = normalized_motion > 2.0
            
            return has_significant_motion, normalized_motion
            
        except Exception as e:
            print(f"[VitalSigns] Motion detection error: {str(e)}")
            return False, 0.0
    
    def _calculate_heart_rate_ensemble(
        self, 
        red_signal: np.ndarray, 
        green_signal: np.ndarray, 
        blue_signal: np.ndarray,
        fps: float = 5.0
    ) -> int:
        """
        Calculate heart rate using ensemble of methods including advanced rPPG algorithms:
        1. POS (Plane-Orthogonal-to-Skin) algorithm
        2. CHROM (Chrominance-based) algorithm
        3. FFT on red channel
        4. FFT on green channel
        5. Autocorrelation
        """
        """
        Calculate heart rate using ensemble of methods:
        1. FFT on red channel (primary)
        2. FFT on green channel (secondary)
        3. Autocorrelation on red channel
        4. Peak detection on filtered signal
        """
        results = []
        weights = []
        method_names = []
        
        # Method 1: POS (Plane-Orthogonal-to-Skin) algorithm (weight: 0.35) - Most robust
        try:
            hr_pos = self._calculate_heart_rate_pos(red_signal, green_signal, blue_signal, fps)
            print(f"[VitalSigns] HR Method 1 (POS): {hr_pos} BPM")
            if 50 <= hr_pos <= 120:
                results.append(hr_pos)
                weights.append(0.35)
                method_names.append('POS')
        except Exception as e:
            print(f"[VitalSigns] HR Method 1 (POS) failed: {str(e)}")
        
        # Method 2: CHROM (Chrominance-based) algorithm (weight: 0.30) - Excellent for motion
        try:
            hr_chrom = self._calculate_heart_rate_chrom(red_signal, green_signal, blue_signal, fps)
            print(f"[VitalSigns] HR Method 2 (CHROM): {hr_chrom} BPM")
            if 50 <= hr_chrom <= 120:
                results.append(hr_chrom)
                weights.append(0.30)
                method_names.append('CHROM')
        except Exception as e:
            print(f"[VitalSigns] HR Method 2 (CHROM) failed: {str(e)}")
        
        # Method 3: FFT on red channel (weight: 0.15)
        try:
            hr_fft_red = self._calculate_heart_rate_temporal(red_signal, fps)
            print(f"[VitalSigns] HR Method 3 (FFT Red): {hr_fft_red} BPM")
            if 50 <= hr_fft_red <= 120:
                results.append(hr_fft_red)
                weights.append(0.15)
                method_names.append('FFT-Red')
        except Exception as e:
            print(f"[VitalSigns] HR Method 3 (FFT Red) failed: {str(e)}")
        
        # Method 4: FFT on green channel (weight: 0.10)
        try:
            hr_fft_green = self._calculate_heart_rate_temporal(green_signal, fps)
            print(f"[VitalSigns] HR Method 4 (FFT Green): {hr_fft_green} BPM")
            if 50 <= hr_fft_green <= 120:
                results.append(hr_fft_green)
                weights.append(0.10)
                method_names.append('FFT-Green')
        except Exception as e:
            print(f"[VitalSigns] HR Method 4 (FFT Green) failed: {str(e)}")
        
        # Method 5: Autocorrelation (weight: 0.10)
        try:
            hr_autocorr = self._calculate_heart_rate_autocorrelation(red_signal, fps)
            print(f"[VitalSigns] HR Method 5 (Autocorr): {hr_autocorr} BPM")
            if 50 <= hr_autocorr <= 120:
                results.append(hr_autocorr)
                weights.append(0.10)
                method_names.append('Autocorr')
        except Exception as e:
            print(f"[VitalSigns] HR Method 5 (Autocorr) failed: {str(e)}")
        
        print(f"[VitalSigns] HR Ensemble results: {results} from methods {method_names}")
        
        if not results:
            print("[VitalSigns] WARNING: All HR methods failed, using fallback value (72 BPM)")
            if fps < 3.0:
                print(f"[VitalSigns] Low FPS ({fps:.2f} Hz) is likely causing detection failure")
                print("[VitalSigns] Recommendation: Capture more frames (15-20) over shorter duration (5-10s)")
            return 72  # Default fallback
        
        # Weighted median (more robust than weighted mean)
        if len(results) == 1:
            return results[0]
        
        # Sort by value and calculate weighted median
        sorted_indices = np.argsort(results)
        sorted_results = [results[i] for i in sorted_indices]
        sorted_weights = [weights[i] for i in sorted_indices]
        cumsum_weights = np.cumsum(sorted_weights)
        
        median_idx = np.searchsorted(cumsum_weights, 0.5)
        final_hr = int(sorted_results[median_idx])
        print(f"[VitalSigns] HR Ensemble final result: {final_hr} BPM (from {len(results)} methods)")
        return final_hr
    
    def _calculate_heart_rate_pos(
        self, 
        red_signal: np.ndarray, 
        green_signal: np.ndarray, 
        blue_signal: np.ndarray,
        fps: float = 5.0
    ) -> int:
        """
        POS (Plane-Orthogonal-to-Skin) algorithm for robust heart rate extraction
        This method projects signals onto a plane orthogonal to skin color, reducing motion artifacts
        """
        if len(red_signal) < 10 or len(green_signal) < 10 or len(blue_signal) < 10:
            return 72
        
        # Normalize signals
        r_mean = np.mean(red_signal)
        g_mean = np.mean(green_signal)
        b_mean = np.mean(blue_signal)
        
        r_norm = red_signal / (r_mean + 1e-6)
        g_norm = green_signal / (g_mean + 1e-6)
        b_norm = blue_signal / (b_mean + 1e-6)
        
        # POS algorithm: Create projection that removes skin color variation
        # The POS signal is orthogonal to the skin color plane
        # Standard POS formula: (R - G) + alpha * (R + G - 2*B)
        # where alpha is a normalization factor
        
        # Calculate alpha based on signal statistics
        rg_diff = r_norm - g_norm
        rgb_sum = r_norm + g_norm - 2 * b_norm
        
        # Adaptive alpha calculation
        if np.std(rgb_sum) > 1e-6:
            alpha = np.std(rg_diff) / np.std(rgb_sum)
        else:
            alpha = 1.0
        
        # POS signal
        pos_signal = rg_diff + alpha * rgb_sum
        
        # Normalize POS signal
        pos_signal = pos_signal - np.mean(pos_signal)
        
        # Apply bandpass filter for heart rate (0.8-3.5 Hz)
        try:
            nyquist = fps / 2.0
            if nyquist >= 0.8:
                low = max(0.8 / nyquist, 0.1)
                high = min(3.5 / nyquist, 0.95)
                if low < high:
                    b, a = butter(4, [low, high], btype='band')
                    pos_signal = filtfilt(b, a, pos_signal)
        except:
            pass
        
        # Calculate heart rate using FFT
        try:
            hr = self._calculate_heart_rate_from_signal(pos_signal, fps, freq_range=(0.8, 3.5))
            if 50 <= hr <= 120:
                return hr
        except:
            pass
        
        return 72
    
    def _calculate_heart_rate_chrom(
        self, 
        red_signal: np.ndarray, 
        green_signal: np.ndarray, 
        blue_signal: np.ndarray,
        fps: float = 5.0
    ) -> int:
        """
        CHROM (Chrominance-based) algorithm for motion-robust heart rate extraction
        Uses chrominance signals that are less affected by motion artifacts
        """
        if len(red_signal) < 10 or len(green_signal) < 10 or len(blue_signal) < 10:
            return 72
        
        # Normalize signals
        r_mean = np.mean(red_signal)
        g_mean = np.mean(green_signal)
        b_mean = np.mean(blue_signal)
        
        r_norm = red_signal / (r_mean + 1e-6)
        g_norm = green_signal / (g_mean + 1e-6)
        b_norm = blue_signal / (b_mean + 1e-6)
        
        # CHROM algorithm: Use chrominance signals X and Y
        # X = R - G (contains pulse information)
        # Y = R + G - 2*B (standard deviation normalization)
        
        X = r_norm - g_norm
        Y = r_norm + g_norm - 2 * b_norm
        
        # Normalize X and Y
        X = X - np.mean(X)
        Y = Y - np.mean(Y)
        
        # Calculate standard deviations for adaptive weighting
        std_X = np.std(X) + 1e-6
        std_Y = np.std(Y) + 1e-6
        
        # CHROM signal: weighted combination
        # The weight adapts based on signal quality
        weight = std_X / (std_X + std_Y)
        chrom_signal = weight * X + (1 - weight) * Y
        
        # Apply bandpass filter for heart rate
        try:
            nyquist = fps / 2.0
            if nyquist >= 0.8:
                low = max(0.8 / nyquist, 0.1)
                high = min(3.5 / nyquist, 0.95)
                if low < high:
                    b, a = butter(4, [low, high], btype='band')
                    chrom_signal = filtfilt(b, a, chrom_signal)
        except:
            pass
        
        # Calculate heart rate using FFT
        try:
            hr = self._calculate_heart_rate_from_signal(chrom_signal, fps, freq_range=(0.8, 3.5))
            if 50 <= hr <= 120:
                return hr
        except:
            pass
        
        return 72
    
    def _calculate_heart_rate_from_signal(
        self, 
        signal: np.ndarray, 
        fps: float, 
        freq_range: Tuple[float, float] = (0.8, 3.5)
    ) -> int:
        """Helper function to calculate HR from filtered signal using FFT"""
        if len(signal) < 10:
            return 72
        
        # Apply windowing
        windowed = signal * np.blackman(len(signal))
        
        # FFT with zero-padding
        fft_size = 2 ** int(np.ceil(np.log2(len(windowed) * 4)))
        fft_result = np.fft.rfft(windowed, n=fft_size)
        freqs = np.fft.rfftfreq(fft_size, 1.0/fps)
        magnitude = np.abs(fft_result)
        
        # Find peak in specified frequency range
        hr_range = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        if not np.any(hr_range):
            return 72
        
        hr_freqs = freqs[hr_range]
        hr_magnitude = magnitude[hr_range]
        
        # Find dominant frequency
        peak_idx = np.argmax(hr_magnitude)
        peak_freq = hr_freqs[peak_idx]
        
        # Convert to BPM
        heart_rate = int(round(peak_freq * 60))
        
        return heart_rate
    
    def _calculate_heart_rate_autocorrelation(self, signal: np.ndarray, fps: float = 10.0) -> int:
        """Calculate heart rate using autocorrelation"""
        if len(signal) < 20:
            return 72
        
        # Normalize
        signal = signal - np.mean(signal)
        
        # Autocorrelation
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation (excluding first peak at lag 0)
        # Heart rate range: 50-120 BPM = 0.83-2.0 Hz
        min_lag = int(fps * 60 / 120)  # 120 BPM
        max_lag = int(fps * 60 / 50)   # 50 BPM
        
        if max_lag >= len(autocorr):
            max_lag = len(autocorr) - 1
        
        if min_lag < max_lag:
            autocorr_range = autocorr[min_lag:max_lag]
            peak_idx = np.argmax(autocorr_range) + min_lag
            
            # Convert lag to heart rate
            period = peak_idx / fps  # seconds
            heart_rate = int(60 / period)
            
            if 50 <= heart_rate <= 120:
                return heart_rate
        
        return 72
    
    def _calculate_heart_rate_peak_detection(self, signal: np.ndarray, fps: float = 10.0) -> int:
        """Calculate heart rate using peak detection"""
        if len(signal) < 20:
            return 72
        
        # Normalize
        signal = signal - np.mean(signal)
        
        # Apply bandpass filter
        try:
            nyquist = fps / 2.0
            low = 0.7 / nyquist
            high = 4.0 / nyquist
            b, a = butter(4, [low, high], btype='band')
            filtered = filtfilt(b, a, signal)
        except:
            filtered = signal
        
        # Find peaks
        # Minimum distance between peaks: 60/120 BPM = 0.5s = fps/2 samples
        min_distance = int(fps * 0.5)
        peaks, properties = find_peaks(filtered, distance=min_distance, prominence=np.std(filtered) * 0.5)
        
        if len(peaks) >= 2:
            # Calculate average interval between peaks
            intervals = np.diff(peaks) / fps  # in seconds
            avg_interval = np.median(intervals)
            heart_rate = int(60 / avg_interval)
            
            if 50 <= heart_rate <= 120:
                return heart_rate
        
        return 72
    
    def _calculate_respiratory_rate_ensemble(self, red_signal: np.ndarray, fps: float = 5.0) -> int:
        """
        Calculate respiratory rate using ensemble of FFT and peak detection
        """
        results = []
        
        # Method 1: FFT (primary)
        try:
            rr_fft = self._calculate_respiratory_rate_temporal(red_signal, fps)
            if 10 <= rr_fft <= 25:
                results.append(rr_fft)
        except:
            pass
        
        # Method 2: Peak detection on low-pass filtered signal
        try:
            rr_peaks = self._calculate_respiratory_rate_peak_detection(red_signal, fps)
            if 10 <= rr_peaks <= 25:
                results.append(rr_peaks)
        except:
            pass
        
        if not results:
            return 16
        
        # Use median for robustness
        return int(np.median(results))
    
    def _calculate_respiratory_rate_peak_detection(self, signal: np.ndarray, fps: float = 10.0) -> int:
        """Calculate respiratory rate using peak detection on low-frequency signal"""
        if len(signal) < 20:
            return 16
        
        # Normalize
        signal = signal - np.mean(signal)
        
        # Apply low-pass filter (0.5 Hz cutoff for respiratory rate)
        try:
            nyquist = fps / 2.0
            high = 0.5 / nyquist
            b, a = butter(4, high, btype='low')
            filtered = filtfilt(b, a, signal)
        except:
            filtered = signal
        
        # Find peaks
        # Minimum distance: 60/30 breaths/min = 2s = fps*2 samples
        min_distance = int(fps * 2)
        peaks, _ = find_peaks(filtered, distance=min_distance, prominence=np.std(filtered) * 0.3)
        
        if len(peaks) >= 2:
            intervals = np.diff(peaks) / fps
            avg_interval = np.median(intervals)
            respiratory_rate = int(60 / avg_interval)
            
            if 10 <= respiratory_rate <= 25:
                return respiratory_rate
        
        return 16
    
    def _get_calibration_factors(self, user_profile: Optional[Dict] = None) -> Dict[str, float]:
        """
        Get calibration factors for SpO2 and temperature based on device/user data
        These can be adjusted based on real-device calibration data
        """
        factors = {
            'spo2_offset': 0.0,  # Offset in percentage points
            'spo2_scale': 1.0,   # Scale factor
            'temp_offset': 0.0,  # Offset in degrees Celsius
            'temp_scale': 1.0    # Scale factor
        }
        
        # Device-specific calibration (can be extended with real calibration data)
        # For now, use default values that can be adjusted per device/user
        
        # Age-based adjustments (older adults may have slightly different SpO2 baselines)
        if user_profile and user_profile.get('age'):
            age = user_profile.get('age')
            if age >= 65:
                factors['spo2_offset'] = -1.0  # Slightly lower baseline for elderly
            elif age < 18:
                factors['spo2_offset'] = 0.5   # Slightly higher for children
        
        # Skin tone adjustments (darker skin may require different calibration)
        # This is a placeholder - real calibration would use device-specific data
        
        return factors
    
    def _calculate_oxygen_saturation_improved(
        self,
        rois: List[bytes],
        red_signal: np.ndarray,
        green_signal: np.ndarray,
        blue_signal: np.ndarray,
        user_profile: Optional[Dict] = None
    ) -> int:
        """
        Improved SpO2 calculation using temporal analysis of R/B ratio
        """
        # Method 1: Traditional R/B ratio from ROIs
        spo2_roi = self._calculate_oxygen_saturation_from_rois(rois)
        
        # Method 2: Temporal R/B ratio analysis
        try:
            if len(red_signal) > 10 and len(blue_signal) > 10:
                # Calculate R/B ratio over time
                rb_ratios = red_signal / (blue_signal + 1e-6)
                
                # Use median ratio for stability
                median_rb = np.median(rb_ratios)
                
                # Optimal R/B ratio for healthy SpO2 is around 1.0-1.2
                optimal_ratio = 1.1
                deviation = abs(median_rb - optimal_ratio)
                
                # Estimate SpO2
                spo2_temporal = 100 - (deviation * 12)
                spo2_temporal = int(np.clip(spo2_temporal, 95, 100))
                
                # Weighted combination: 60% ROI method, 40% temporal
                spo2_combined = int(0.6 * spo2_roi + 0.4 * spo2_temporal)
                
                # Apply calibration factors
                calibration = self._get_calibration_factors(user_profile)
                spo2_combined = int((spo2_combined * calibration['spo2_scale']) + calibration['spo2_offset'])
                
                return int(np.clip(spo2_combined, 70, 100))  # Wider range to allow for calibration
        except:
            pass
        
        # Apply calibration to ROI method result as well
        calibration = self._get_calibration_factors(user_profile)
        spo2_roi_calibrated = int((spo2_roi * calibration['spo2_scale']) + calibration['spo2_offset'])
        return int(np.clip(spo2_roi_calibrated, 70, 100))

    def _estimate_temperature(self, image: np.ndarray, sensor_data: Optional[Dict] = None) -> float:
        """
        Estimate facial skin temperature proxy using color intensity and optional sensor hints
        """
        baseline = 36.5

        if sensor_data:
            external_temp = sensor_data.get('ambientTemperature') or sensor_data.get('temperature')
            if isinstance(external_temp, (int, float)):
                baseline = float(external_temp)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        red_mean = np.mean(rgb[:, :, 0])
        green_mean = np.mean(rgb[:, :, 1])

        perfusion_index = (red_mean - green_mean) / (green_mean + 1e-6)
        skin_adjustment = np.clip(perfusion_index * 1.5, -0.5, 1.0)

        estimate = baseline + skin_adjustment
        return float(np.clip(estimate, 35.5, 38.5))

    def _estimate_temperature_from_rois(
        self,
        rois: List[bytes],
        sensor_data: Optional[Dict] = None,
        user_profile: Optional[Dict] = None
    ) -> float:
        temps = []
        for roi_bytes in rois:
            try:
                image = decode_image_bytes(roi_bytes)
                if image is not None:
                    temps.append(self._estimate_temperature(image, sensor_data))
            except Exception:
                continue

        if temps:
            return float(np.median(temps))

        if sensor_data:
            external_temp = sensor_data.get('ambientTemperature') or sensor_data.get('temperature')
            if isinstance(external_temp, (int, float)):
                return float(np.clip(external_temp, 35.5, 38.5))

        return 36.5
    
    def _estimate_blood_pressure(self, heart_rate: int, stress_level: int, signal: np.ndarray) -> Dict[str, int]:
        """
        Estimate blood pressure based on heart rate, stress level, and signal characteristics
        This is a simplified estimation - actual BP requires specialized equipment
        
        Args:
            heart_rate: Heart rate in BPM
            stress_level: Stress level 0-100
            signal: Signal array for additional analysis
            
        Returns:
            Dict with systolic and diastolic BP in mmHg
        """
        # Base BP values (normal: 120/80)
        base_systolic = 120
        base_diastolic = 80
        
        # Adjust based on heart rate (higher HR may indicate higher BP)
        hr_factor = (heart_rate - 70) / 20.0  # Normalize around 70 BPM
        hr_adjustment = hr_factor * 5  # Max ±5 mmHg per 20 BPM difference
        
        # Adjust based on stress level (stress can elevate BP)
        stress_factor = (stress_level - 50) / 50.0  # Normalize around 50
        stress_adjustment = stress_factor * 10  # Max ±10 mmHg
        
        # Calculate signal variability (higher variability may indicate BP changes)
        signal_variability = np.std(signal) if len(signal) > 0 else 0
        var_adjustment = (signal_variability - 1.0) * 2  # Small adjustment based on variability
        
        # Calculate final BP
        systolic = int(np.clip(base_systolic + hr_adjustment + stress_adjustment + var_adjustment, 90, 180))
        diastolic = int(np.clip(base_diastolic + hr_adjustment * 0.6 + stress_adjustment * 0.6 + var_adjustment * 0.6, 60, 120))
        
        return {
            'systolic': systolic,
            'diastolic': diastolic
        }
    
    def _default_vitals(self) -> Dict:
        """Return default vital signs values"""
        return {
            'heartRate': 0,
            'stressLevel': 0,
            'oxygenSaturation': 0,
            'respiratoryRate': 0,
            'temperature': 0,
            'bloodPressure': {'systolic': 0, 'diastolic': 0}
        }

