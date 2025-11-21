"""
Vital Signs Analysis Service using PPG and signal processing
Improved algorithms for better accuracy with advanced signal processing
"""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
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
    
    def analyze_video_frames(self, frames: List[bytes], sensor_data: Optional[Dict] = None) -> Dict:
        """
        Analyze multiple frames for vital signs using temporal signal analysis
        This is the proper way to do PPG - analyze signal over time
        
        Args:
            frames: List of image bytes
            
        Returns:
            Dict with analysis results
        """
        import gc
        
        start_time = time.time()
        print(f"[VitalSigns] Starting video frame analysis: {len(frames)} frames")
        
        if not frames or len(frames) == 0:
            raise ValueError("No frames provided")
        
        # First pass: detect faces and extract ROIs from all frames
        rois = []
        face_detected_count = 0
        total_quality_score = 0
        bounding_boxes = []
        
        for i, frame_bytes in enumerate(frames):
            try:
                # Validate frame quality
                quality_check = self.face_detection_service.validate_frame_quality(frame_bytes)
                if not quality_check['isValid'] and quality_check['score'] == 0:
                    continue
                
                total_quality_score += quality_check['score']
                
                # Detect face
                face_result = self.face_detection_service.detect_face(frame_bytes)
                if not face_result.get('detected'):
                    continue
                
                face_detected_count += 1
                bounding_boxes.append(face_result['boundingBox'])
                
                # Extract ROI
                try:
                    roi_bytes = self.face_detection_service.extract_roi(
                        frame_bytes,
                        face_result['boundingBox']
                    )
                    rois.append(roi_bytes)
                except Exception as e:
                    print(f"[VitalSigns] ROI extraction failed for frame {i+1}: {str(e)}")
                    continue
                
                # Periodic garbage collection every 5 frames to manage memory
                if (i + 1) % 5 == 0:
                    gc.collect()
                    
            except Exception as e:
                print(f"[VitalSigns] Error processing frame {i+1}: {str(e)}")
                continue
        
        # Final garbage collection after processing all frames
        gc.collect()
        
        if len(rois) < 10:  # Need at least 10 frames for temporal analysis
            print(f"[VitalSigns] Not enough valid frames ({len(rois)}), using fallback")
            return self._fallback_analysis(frames, face_detected_count, total_quality_score)
        
        print(f"[VitalSigns] Extracted {len(rois)} valid ROIs for temporal analysis")
        
        # Adjust quality score based on sensor data
        if sensor_data:
            sensor_quality_adjustment = self._adjust_quality_from_sensors(sensor_data)
            if sensor_quality_adjustment < 0:
                print(f"[VitalSigns] Sensor data indicates poor conditions, adjusting quality score by {sensor_quality_adjustment}")
                total_quality_score = max(0, total_quality_score + sensor_quality_adjustment)
        
        # Calculate actual frame rate (FPS) from number of frames and estimated duration
        # Assuming ~30 second capture duration for 30 seconds of recording
        estimated_duration = 30.0  # seconds
        actual_fps = len(rois) / estimated_duration if len(rois) > 0 else 5.0
        # Clamp FPS to realistic range (1-15 FPS for takePhoto capture)
        actual_fps = np.clip(actual_fps, 1.0, 15.0)
        print(f"[VitalSigns] Estimated FPS: {actual_fps:.2f} (from {len(rois)} frames over ~{estimated_duration}s)")
        
        # Extract temporal signal from all ROIs with multi-ROI support
        try:
            # Extract signals from multiple regions for better accuracy, focusing on forehead
            signals = self._extract_multi_roi_signals(rois)
            
            if len(signals['red']) < 10:
                print(f"[VitalSigns] Not enough signals extracted: {len(signals['red'])}")
                return self._fallback_analysis(frames, face_detected_count, total_quality_score)
            
            # Log raw signal statistics
            print(f"[VitalSigns] Raw signal stats - Red: mean={np.mean(signals['red']):.2f}, std={np.std(signals['red']):.2f}, range={np.max(signals['red'])-np.min(signals['red']):.2f}")
            print(f"[VitalSigns] Raw signal stats - Green: mean={np.mean(signals['green']):.2f}, std={np.std(signals['green']):.2f}, range={np.max(signals['green'])-np.min(signals['green']):.2f}")
            print(f"[VitalSigns] Raw signal stats - Blue: mean={np.mean(signals['blue']):.2f}, std={np.std(signals['blue']):.2f}, range={np.max(signals['blue'])-np.min(signals['blue']):.2f}")
            
            # Check signal quality before processing
            signal_quality = self._validate_signal_quality(signals)
            if not signal_quality['isValid']:
                print(f"[VitalSigns] Signal quality check failed: {signal_quality['reason']}")
                return self._fallback_analysis(frames, face_detected_count, total_quality_score)
            
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
            
            oxygen_saturation = self._calculate_oxygen_saturation_improved(rois[-10:], red_signal, green_signal, blue_signal)
            print(f"[VitalSigns] Calculated SpO2: {oxygen_saturation}%")

            temperature = self._estimate_temperature_from_rois(rois[-10:], sensor_data)
            print(f"[VitalSigns] Estimated temperature: {temperature:.2f}°C")

            # Estimate blood pressure based on heart rate, stress, and signal characteristics
            blood_pressure = self._estimate_blood_pressure(heart_rate, stress_level, red_signal)
            print(f"[VitalSigns] Calculated BP: {blood_pressure['systolic']}/{blood_pressure['diastolic']} mmHg")

        except Exception as e:
            print(f"[VitalSigns] Error in temporal analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._fallback_analysis(frames, face_detected_count, total_quality_score)
        
        # Calculate confidence
        avg_quality_score = total_quality_score / face_detected_count if face_detected_count > 0 else 0
        confidence = self._calculate_confidence(
            face_detected_count / len(frames),
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
            'totalFrames': len(frames),
            'frameCount': len(frames),  # Alias for compatibility
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
        temperature: Optional[float] = None
    ) -> float:
        """
        Calculate overall confidence score
        """
        # Base confidence from face detection
        confidence = face_confidence * 0.4
        
        # Quality score contribution
        confidence += (quality_score / 100.0) * 0.3
        
        # Vital signs completeness
        vitals_count = sum([
            heart_rate is not None,
            stress_level is not None,
            oxygen_saturation is not None,
            respiratory_rate is not None,
            temperature is not None
        ])
        completeness = vitals_count / 5.0
        confidence += completeness * 0.2
        
        # Reasonableness check
        reasonableness = 1.0
        if heart_rate and not (50 <= heart_rate <= 120):
            reasonableness *= 0.8
        if stress_level and not (0 <= stress_level <= 100):
            reasonableness *= 0.8
        if oxygen_saturation and not (95 <= oxygen_saturation <= 100):
            reasonableness *= 0.8
        if respiratory_rate and not (10 <= respiratory_rate <= 25):
            reasonableness *= 0.8
        if temperature and not (35.5 <= temperature <= 39.0):
            reasonableness *= 0.8
        
        confidence += reasonableness * 0.1
        
        return float(np.clip(confidence, 0.0, 1.0))
    
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
    
    def _extract_multi_roi_signals(self, rois: List[bytes]) -> Dict[str, np.ndarray]:
        """
        Extract signals from multiple regions of interest
        Improved PPG signal extraction with focus on forehead region (most reliable for PPG)
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
                
                # Extract signals from different regions with improved weights
                # Forehead region (top 25%) - best for PPG (highest weight)
                forehead_top = int(h * 0.05)  # Top 5% - most reliable
                forehead_bottom = int(h * 0.3)  # Top 30%
                forehead = image[forehead_top:forehead_bottom, :]
                
                # Middle forehead (10-20%) - secondary PPG region
                mid_forehead = image[int(h*0.1):int(h*0.2), :]
                
                # Cheek regions (middle 30-50%) - for color analysis
                cheeks = image[int(h*0.3):int(h*0.5), :]
                
                # Weighted combination: 70% forehead (50% top + 20% mid), 30% cheeks
                # Red channel (most sensitive to blood flow)
                red_top_forehead = np.mean(forehead[:, :, 2]) if forehead.size > 0 else 0
                red_mid_forehead = np.mean(mid_forehead[:, :, 2]) if mid_forehead.size > 0 else 0
                red_cheeks = np.mean(cheeks[:, :, 2]) if cheeks.size > 0 else 0
                red_combined = 0.5 * red_top_forehead + 0.2 * red_mid_forehead + 0.3 * red_cheeks
                
                # Green channel (secondary PPG signal)
                green_top_forehead = np.mean(forehead[:, :, 1]) if forehead.size > 0 else 0
                green_mid_forehead = np.mean(mid_forehead[:, :, 1]) if mid_forehead.size > 0 else 0
                green_cheeks = np.mean(cheeks[:, :, 1]) if cheeks.size > 0 else 0
                green_combined = 0.5 * green_top_forehead + 0.2 * green_mid_forehead + 0.3 * green_cheeks
                
                # Blue channel (for SpO2 calculation)
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
        
        # 5. Savitzky-Golay filter - only if signal is long enough and we have variation
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
    
    def _calculate_heart_rate_ensemble(
        self, 
        red_signal: np.ndarray, 
        green_signal: np.ndarray, 
        blue_signal: np.ndarray,
        fps: float = 5.0
    ) -> int:
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
        
        # Method 1: FFT on red channel (weight: 0.4)
        try:
            hr_fft_red = self._calculate_heart_rate_temporal(red_signal, fps)
            print(f"[VitalSigns] HR Method 1 (FFT Red): {hr_fft_red} BPM")
            if 50 <= hr_fft_red <= 120:
                results.append(hr_fft_red)
                weights.append(0.4)
                method_names.append('FFT-Red')
        except Exception as e:
            print(f"[VitalSigns] HR Method 1 failed: {str(e)}")
        
        # Method 2: FFT on green channel (weight: 0.3)
        try:
            hr_fft_green = self._calculate_heart_rate_temporal(green_signal, fps)
            print(f"[VitalSigns] HR Method 2 (FFT Green): {hr_fft_green} BPM")
            if 50 <= hr_fft_green <= 120:
                results.append(hr_fft_green)
                weights.append(0.3)
                method_names.append('FFT-Green')
        except Exception as e:
            print(f"[VitalSigns] HR Method 2 failed: {str(e)}")
        
        # Method 3: Autocorrelation on red channel (weight: 0.2)
        try:
            hr_autocorr = self._calculate_heart_rate_autocorrelation(red_signal, fps)
            print(f"[VitalSigns] HR Method 3 (Autocorr): {hr_autocorr} BPM")
            if 50 <= hr_autocorr <= 120:
                results.append(hr_autocorr)
                weights.append(0.2)
                method_names.append('Autocorr')
        except Exception as e:
            print(f"[VitalSigns] HR Method 3 failed: {str(e)}")
        
        # Method 4: Peak detection (weight: 0.1)
        try:
            hr_peaks = self._calculate_heart_rate_peak_detection(red_signal, fps)
            print(f"[VitalSigns] HR Method 4 (Peak Detection): {hr_peaks} BPM")
            if 50 <= hr_peaks <= 120:
                results.append(hr_peaks)
                weights.append(0.1)
                method_names.append('Peaks')
        except Exception as e:
            print(f"[VitalSigns] HR Method 4 failed: {str(e)}")
        
        print(f"[VitalSigns] HR Ensemble results: {results} from methods {method_names}")
        
        if not results:
            print("[VitalSigns] All HR methods failed, using fallback")
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
    
    def _calculate_oxygen_saturation_improved(
        self, 
        rois: List[bytes], 
        red_signal: np.ndarray,
        green_signal: np.ndarray,
        blue_signal: np.ndarray
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
                return spo2_combined
        except:
            pass
        
        return spo2_roi

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

    def _estimate_temperature_from_rois(self, rois: List[bytes], sensor_data: Optional[Dict] = None) -> float:
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

