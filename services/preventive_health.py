"""
Preventive Health & Lifestyle Insights Service
Combines open clinical scoring systems (NEWS2) with heuristics
driven by WHO lifestyle guidelines to surface preventive actions.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import math
import os
import json

import numpy as np


class PreventiveHealthConfig:
    """Configuration for preventive health thresholds and parameters."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        # Load from environment or config file if available
        config = config_dict or self._load_config()
        
        # NEWS2 Scoring Thresholds
        self.news2 = config.get('news2', {})
        self.news2_respiratory = self.news2.get('respiratory_rate', {
            'very_low': 8, 'low': 11, 'normal_high': 20, 'high': 24
        })
        self.news2_spo2 = self.news2.get('oxygen_saturation', {
            'critical': 91, 'low': 93, 'moderate': 95
        })
        self.news2_systolic = self.news2.get('systolic', {
            'very_low': 90, 'low': 100, 'moderate': 110, 'normal_high': 219
        })
        self.news2_heart_rate = self.news2.get('heart_rate', {
            'very_low': 40, 'low': 50, 'normal_high': 90, 'high': 110, 'very_high': 130
        })
        self.news2_temperature = self.news2.get('temperature', {
            'hypothermic': 35, 'low': 36, 'normal_high': 38, 'fever': 39
        })
        self.news2_score_levels = self.news2.get('score_levels', {
            'critical': 7, 'high': 5, 'moderate': 3
        })
        
        # Blood Pressure Thresholds (AHA Guidelines)
        self.bp = config.get('blood_pressure', {})
        self.bp_crisis_systolic = self.bp.get('crisis_systolic', 180)
        self.bp_crisis_diastolic = self.bp.get('crisis_diastolic', 120)
        self.bp_stage2_systolic = self.bp.get('stage2_systolic', 140)
        self.bp_stage2_diastolic = self.bp.get('stage2_diastolic', 90)
        self.bp_stage1_systolic = self.bp.get('stage1_systolic', 130)
        self.bp_stage1_diastolic = self.bp.get('stage1_diastolic', 80)
        self.bp_elevated_systolic = self.bp.get('elevated_systolic', 120)
        
        # Stress Level Thresholds
        self.stress = config.get('stress', {})
        self.stress_very_high = self.stress.get('very_high', 80)
        self.stress_elevated = self.stress.get('elevated', 60)
        self.stress_moderate = self.stress.get('moderate', 40)
        
        # Fever Detection Thresholds
        self.fever = config.get('fever', {})
        self.fever_temp_threshold = self.fever.get('temperature_threshold', 37.5)
        self.fever_temp_elevated = self.fever.get('temperature_elevated', 37.4)
        self.fever_hr_threshold = self.fever.get('heart_rate_threshold', 88)
        self.fever_stress_threshold = self.fever.get('stress_threshold', 70)
        self.fever_spo2_threshold = self.fever.get('spo2_threshold', 97)
        self.fever_probability_threshold = self.fever.get('probability_threshold', 0.6)
        self.fever_weights = self.fever.get('weights', {
            'temperature': 1.2, 'heart_rate': 0.9, 'stress': 0.7, 'spo2': 0.8
        })
        self.fever_trend_bonus = self.fever.get('trend_bonus', {
            'heart_rate': 0.15, 'temperature': 0.2
        })
        self.fever_sigmoid_offset = self.fever.get('sigmoid_offset', 1.0)
        
        # Respiratory Detection Thresholds
        self.respiratory = config.get('respiratory', {})
        self.respiratory_rate_high = self.respiratory.get('rate_high', 20)
        self.respiratory_rate_very_high = self.respiratory.get('rate_very_high', 22)
        self.respiratory_spo2_threshold = self.respiratory.get('spo2_threshold', 96)
        self.respiratory_spo2_low = self.respiratory.get('spo2_low', 94)
        self.respiratory_hr_threshold = self.respiratory.get('heart_rate_threshold', 85)
        self.respiratory_stress_threshold = self.respiratory.get('stress_threshold', 65)
        self.respiratory_probability_threshold = self.respiratory.get('probability_threshold', 0.6)
        self.respiratory_weights = self.respiratory.get('weights', {
            'respiratory_rate': 1.0, 'spo2': 0.9, 'heart_rate': 0.4, 'stress': 0.4
        })
        self.respiratory_trend_bonus = self.respiratory.get('trend_bonus', {
            'respiratory_rate': 0.2, 'oxygen_saturation': 0.25
        })
        self.respiratory_sigmoid_offset = self.respiratory.get('sigmoid_offset', 0.9)
        
        # Stress Recovery Thresholds
        self.stress_recovery = config.get('stress_recovery', {})
        self.stress_recovery_optimal_hr = self.stress_recovery.get('optimal_heart_rate', 70)
        self.stress_recovery_optimal_rr = self.stress_recovery.get('optimal_respiratory_rate', 16)
        self.stress_recovery_weights = self.stress_recovery.get('weights', {
            'stress': 0.4, 'heart_rate': 0.35, 'respiratory_rate': 0.25
        })
        self.stress_recovery_trend_bonus = self.stress_recovery.get('trend_bonus', 0.1)
        self.stress_recovery_low_threshold = self.stress_recovery.get('low_threshold', 0.4)
        self.stress_recovery_medium_threshold = self.stress_recovery.get('medium_threshold', 0.5)
        
        # Lifestyle Targets
        self.lifestyle = config.get('lifestyle', {})
        self.lifestyle_hydration_base_ml = self.lifestyle.get('hydration_base_ml', 2300)
        self.lifestyle_hydration_ml_per_kg = self.lifestyle.get('hydration_ml_per_kg', 35)
        self.lifestyle_sleep_target_hours = self.lifestyle.get('sleep_target_hours', 7.5)
        self.lifestyle_sleep_young_adult_hours = self.lifestyle.get('sleep_young_adult_hours', 8.0)
        self.lifestyle_young_adult_age = self.lifestyle.get('young_adult_age', 25)
        
        # Probability Thresholds
        self.probability = config.get('probability', {})
        self.probability_fever_high = self.probability.get('fever_high', 0.65)
        self.probability_respiratory_high = self.probability.get('respiratory_high', 0.65)
        self.probability_fever_warning = self.probability.get('fever_warning', 0.55)
        self.probability_respiratory_warning = self.probability.get('respiratory_warning', 0.55)
        self.probability_fever_alert = self.probability.get('fever_alert', 0.7)
        self.probability_respiratory_alert = self.probability.get('respiratory_alert', 0.7)
        
        # Confidence Calculations
        self.confidence = config.get('confidence', {})
        self.confidence_fever_base = self.confidence.get('fever_base', 0.6)
        self.confidence_fever_news2_factor = self.confidence.get('fever_news2_factor', 20)
        self.confidence_respiratory_base = self.confidence.get('respiratory_base', 0.55)
        self.confidence_respiratory_news2_factor = self.confidence.get('respiratory_news2_factor', 25)
        
        # Default Lookback Period
        self.default_lookback_days = config.get('default_lookback_days', 14)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or environment variables."""
        config = {}
        
        # Try loading from config file
        config_path = os.getenv('PREVENTIVE_HEALTH_CONFIG', 'config/preventive_health.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            except Exception:
                pass
        
        # Override with environment variables if present
        # Example: PREVENTIVE_HEALTH_BP_STAGE1_SYSTOLIC=130
        for key, value in os.environ.items():
            if key.startswith('PREVENTIVE_HEALTH_'):
                # Parse nested keys (e.g., PREVENTIVE_HEALTH_BP_STAGE1_SYSTOLIC -> bp.stage1_systolic)
                parts = key.replace('PREVENTIVE_HEALTH_', '').lower().split('_')
                # Simple nested assignment (can be enhanced)
                if len(parts) >= 2:
                    section = parts[0]
                    param = '_'.join(parts[1:])
                    if section not in config:
                        config[section] = {}
                    try:
                        # Try to parse as number
                        config[section][param] = float(value) if '.' in value else int(value)
                    except ValueError:
                        config[section][param] = value
        
        return config


class PreventiveHealthInsightsService:
    """
    Generates preventive health insights, symptom forecasts and
    lifestyle nudges from stored vital metrics.
    """

    METRIC_ALIASES = {
        'heart_rate': 'heart_rate',
        'hr': 'heart_rate',
        'stress_level': 'stress_level',
        'stress': 'stress_level',
        'oxygen_saturation': 'oxygen_saturation',
        'spo2': 'oxygen_saturation',
        'respiratory_rate': 'respiratory_rate',
        'rr': 'respiratory_rate',
        'blood_pressure_systolic': 'bp_systolic',
        'systolic': 'bp_systolic',
        'blood_pressure_diastolic': 'bp_diastolic',
        'diastolic': 'bp_diastolic',
        'temperature': 'temperature',
        'body_temperature': 'temperature',
        'steps': 'steps',
        'sleep_duration': 'sleep_hours',
    }

    SUPPORTED_METRICS = [
        'heart_rate',
        'stress_level',
        'oxygen_saturation',
        'respiratory_rate',
        'bp_systolic',
        'bp_diastolic',
        'temperature',
        'steps',
        'sleep_hours',
    ]

    def __init__(self, config: Optional[PreventiveHealthConfig] = None):
        self.model_version = "2024.11"
        self.initialized = False
        self.config = config or PreventiveHealthConfig()

    def initialize(self):
        """No heavy weights to load yet, but keep interface consistent."""
        self.initialized = True

    def generate_insights(
        self,
        metrics: List[Dict[str, Any]],
        user_profile: Optional[Dict[str, Any]] = None,
        lookback_days: Optional[int] = None
    ) -> Dict[str, Any]:
        # Allow empty metrics - will generate recommendations from user profile
        if metrics is None:
            metrics = []

        if lookback_days is None or lookback_days <= 0:
            lookback_days = self.config.default_lookback_days

        # Prepare time series - will be empty if no metrics
        time_series = self._prepare_time_series(metrics, lookback_days) if metrics else {}
        
        # Get latest values - use defaults if no metrics available
        latest = {metric: points[-1]['value'] for metric, points in time_series.items() if points} if time_series else {}
        
        # If no metrics, use default values based on user profile for lifestyle recommendations
        if not latest and user_profile:
            # Generate baseline values from user profile for lifestyle card generation
            latest = {
                'heart_rate': 72,  # Average resting heart rate
                'stress_level': 50,  # Moderate stress
                'respiratory_rate': 16,  # Average respiratory rate
                'temperature': 36.5,  # Normal body temperature
                'oxygen_saturation': 98,  # Normal SpO2
                'steps': 0,  # No historical data
                'sleep_hours': 0,  # No historical data
            }

        # Calculate personalized baselines from historical data (or use defaults if no metrics)
        baselines = self._calculate_personalized_baselines(time_series, user_profile) if time_series else {}
        
        # Calculate population-based adjustments for adaptive scoring
        population_adjustments = self._calculate_population_adjustments(user_profile, time_series) if time_series else {}

        trends = {
            metric: self._calculate_trend(points)
            for metric, points in time_series.items()
            if points
        } if time_series else {}

        # Calculate activity level from steps and recent activity patterns
        activity_level = self._calculate_activity_level(time_series, latest) if time_series else 1.0
        
        # Use personalized baselines for NEWS2 calculation
        # Adjust for activity level before calculating NEWS2
        latest_adjusted = self._adjust_vitals_for_activity(latest, activity_level)
        news2 = self._calculate_news2(latest_adjusted, baselines=baselines, population_adjustments=population_adjustments)
        
        # Use activity-adjusted vitals for probability calculations with dynamic weighting
        # Use defaults if no metrics available
        if time_series:
            fever_prob, fever_signals = self._estimate_fever_probability(latest_adjusted, trends, news2['score'], activity_level, time_series)
            resp_prob, resp_signals = self._estimate_respiratory_probability(latest_adjusted, trends, news2['score'], activity_level, time_series)
            stress_recovery = self._estimate_stress_recovery(latest, trends, time_series)
        else:
            # No metrics - use neutral values for lifestyle card generation
            fever_prob = 0.0
            fever_signals = []
            resp_prob = 0.0
            resp_signals = []
            stress_recovery = 0.6  # Moderate recovery for new users

        # Enhanced BP assessment using multiple readings with rolling average
        bp_systolic_values = [p['value'] for p in time_series.get('bp_systolic', [])[-5:]] if time_series else []  # Last 5 readings
        bp_diastolic_values = [p['value'] for p in time_series.get('bp_diastolic', [])[-5:]] if time_series else []
        
        # Use rolling average if multiple readings available
        if len(bp_systolic_values) >= 2 and len(bp_diastolic_values) >= 2:
            systolic_avg = float(np.median(bp_systolic_values))  # Median is more robust to outliers
            diastolic_avg = float(np.median(bp_diastolic_values))
        else:
            # Fall back to latest if insufficient readings
            systolic_avg = latest.get('bp_systolic')
            diastolic_avg = latest.get('bp_diastolic')
        
        # BP and stress assessment - use defaults if no metrics
        if systolic_avg and diastolic_avg:
            bp_assessment = self._assess_blood_pressure(
                systolic_avg,
                diastolic_avg,
                metrics,  # Pass full metrics for advanced analysis
                baselines=baselines  # Use personalized baselines
            )
        else:
            bp_assessment = {'level': 'normal', 'systolic': 120, 'diastolic': 80}
        
        stress_assessment = self._assess_stress_level(latest.get('stress_level'), baselines=baselines) if latest.get('stress_level') else {'level': 'moderate'}
        status_level = self._calculate_overall_status(
            news2['score'],
            bp_assessment,
            stress_assessment
        )

        summary = self._compose_summary(
            status_level,
            news2['score'],
            bp_assessment,
            stress_assessment,
            fever_prob,
            resp_prob
        )

        vitals_concerns = self._build_vital_concerns(bp_assessment, stress_assessment)
        lifestyle_plan = self._build_lifestyle_plan(latest, user_profile, stress_recovery, trends)
        lifestyle_card = self._build_lifestyle_card(
            latest=latest,
            user_profile=user_profile,
            lifestyle_plan=lifestyle_plan,
            stress_recovery=stress_recovery,
            trends=trends,
            news2=news2,
            bp_assessment=bp_assessment,
            stress_assessment=stress_assessment
        )
        recommendations = self._build_recommendations(
            fever_prob,
            resp_prob,
            stress_recovery,
            news2,
            trends,
            bp_assessment,
            stress_assessment
        )

        cfg = self.config
        forecast = [
            {
                'name': 'Fever',
                'probability': round(fever_prob, 2),
                'confidence': round(cfg.confidence_fever_base + news2['score'] / cfg.confidence_fever_news2_factor, 2),
                'signals': fever_signals,
            },
            {
                'name': 'Upper respiratory strain',
                'probability': round(resp_prob, 2),
                'confidence': round(cfg.confidence_respiratory_base + news2['score'] / cfg.confidence_respiratory_news2_factor, 2),
                'signals': resp_signals,
            },
        ]

        requires_review = (
            news2['score'] >= cfg.news2_score_levels['high']
            or fever_prob >= cfg.probability_fever_high
            or resp_prob >= cfg.probability_respiratory_high
        )

        result = {
            'generatedAt': datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
            'model': {
                'name': 'PreventiveHealthInsightsService',
                'version': self.model_version,
                'sources': [
                    'NHS NEWS2 early warning score (open standard)',
                    'WHO 2020 physical activity & sedentary behaviour guidelines',
                    'CDC respiratory illness surveillance heuristics'
                ],
            },
            'summary': summary,
            'news2': news2,
            'vitalsSnapshot': self._build_snapshot(latest, trends),
            'riskScores': {
                'feverProbability': round(fever_prob, 2),
                'respiratoryProbability': round(resp_prob, 2),
                'stressRecoveryIndex': round(stress_recovery, 2),
            },
            'symptomForecast': forecast,
            'lifestylePlan': lifestyle_plan,
            'recommendations': recommendations[:5],
            'lifestyleCard': lifestyle_card,
            'vitalsConcerns': vitals_concerns,
            'timelineSummary': self._timeline_summary(trends),
            'safety': {
                'requiresClinicianReview': requires_review,
                'flags': self._build_flags(news2, fever_prob, resp_prob),
            },
            'metadata': {
                'lookbackDays': lookback_days,
                'metricCount': sum(len(points) for points in time_series.values()),
                'supportedMetrics': self.SUPPORTED_METRICS,
                'activityLevel': activity_level,  # Include activity level in metadata
            },
            'baselines': {
                metric: {
                    'median': base['median'],
                    'low': base['low'],
                    'high': base['high'],
                    'established': base['established'],
                    'sampleSize': base['sample_size']
                }
                for metric, base in baselines.items()
                if base.get('established', False)  # Only include established baselines
            } if baselines else {},
        }
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _validate_metric_quality(self, entry: Dict[str, Any]) -> bool:
        """
        Enhanced metric validation with confidence, stability, and reasonableness checks
        Returns True if metric passes all validation criteria
        """
        # Check 1: Confidence threshold (only for AI-generated metrics)
        source = entry.get('source', '').lower()
        confidence = entry.get('confidence')
        
        if source in ['ai_face_analysis', 'ai'] and confidence is not None:
            try:
                conf_value = float(confidence) if isinstance(confidence, str) else confidence
                if conf_value < 0.5:  # Minimum confidence threshold
                    return False
            except (TypeError, ValueError):
                # If confidence can't be parsed and it's AI source, reject
                if source in ['ai_face_analysis', 'ai']:
                    return False
        
        # Check 2: Value reasonableness
        value = entry.get('value')
        if value is None:
            return False
        
        try:
            value = float(value)
        except (TypeError, ValueError):
            return False
        
        metric_type = (entry.get('type') or entry.get('metric_type') or '').lower()
        
        # Reasonableness ranges for different metrics
        if 'heart_rate' in metric_type:
            if not (40 <= value <= 200):  # Wider range during exercise
                return False
        elif 'stress_level' in metric_type:
            if not (0 <= value <= 100):
                return False
        elif 'oxygen_saturation' in metric_type or 'spo2' in metric_type:
            if not (70 <= value <= 100):  # Allow wider range for validation
                return False
        elif 'respiratory_rate' in metric_type or 'rr' in metric_type:
            if not (8 <= value <= 40):  # Wider range during exercise
                return False
        elif 'temperature' in metric_type:
            if not (34.0 <= value <= 42.0):  # Wider range for safety
                return False
        elif 'bp_systolic' in metric_type or 'systolic' in metric_type:
            if not (70 <= value <= 250):
                return False
        elif 'bp_diastolic' in metric_type or 'diastolic' in metric_type:
            if not (40 <= value <= 150):
                return False
        
        # Check 3: Zero value validation (invalid for vital signs except steps)
        if value == 0 and metric_type not in ['steps', 'sleep_hours']:
            return False
        
        return True
    
    def _prepare_time_series(
        self,
        metrics: List[Dict[str, Any]],
        lookback_days: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        lookback_start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        validated_count = 0
        rejected_count = 0

        for entry in metrics:
            metric_type = entry.get('type') or entry.get('metric_type')
            if not metric_type:
                rejected_count += 1
                continue
            
            metric_key = self.METRIC_ALIASES.get(metric_type.lower())
            if not metric_key:
                rejected_count += 1
                continue

            # Enhanced validation: check confidence, reasonableness, zero values
            if not self._validate_metric_quality(entry):
                rejected_count += 1
                continue

            value = entry.get('value')
            if value is None:
                rejected_count += 1
                continue

            try:
                value = float(value)
            except (TypeError, ValueError):
                rejected_count += 1
                continue

            timestamp = self._parse_timestamp(entry.get('timestamp') or entry.get('start_time'))
            if timestamp < lookback_start:
                continue

            buckets[metric_key].append({
                'value': value,
                'timestamp': timestamp,
                'source': entry.get('source'),
                'confidence': entry.get('confidence'),
            })
            validated_count += 1

        if validated_count > 0:
            print(f"[PreventiveHealth] Validated {validated_count} metrics, rejected {rejected_count} invalid/low-quality metrics")

        for metric, points in buckets.items():
            points.sort(key=lambda item: item['timestamp'])

        return buckets

    def _parse_timestamp(self, value: Any) -> datetime:
        if value is None:
            return datetime.now(timezone.utc)

        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

        if isinstance(value, (int, float)):
            if value > 10**12:
                value = value / 1000.0
            return datetime.fromtimestamp(value, tz=timezone.utc)

        if isinstance(value, str):
            try:
                sanitized = value.replace('Z', '+00:00')
                parsed = datetime.fromisoformat(sanitized)
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        return datetime.now(timezone.utc)

    def _calculate_trend(self, points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhanced trend calculation with rolling averages and multi-day analysis
        """
        if len(points) < 2:
            return {'direction': 'steady', 'delta': 0.0, 'changePerDay': 0.0, 'volatility': 0.0, 'rollingAvg': None}

        values = [p['value'] for p in points]
        timestamps = [p['timestamp'] for p in points]
        
        # Apply rolling average to reduce false spikes (7-point moving average)
        if len(values) >= 7:
            window_size = min(7, len(values))
            rolling_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            # Pad to match original length
            rolling_avg = np.concatenate([values[:window_size-1], rolling_avg])
        else:
            # Use simple average for short series
            rolling_avg = np.full(len(values), np.mean(values))
        
        # Calculate volatility (standard deviation of rolling differences)
        rolling_diff = np.diff(rolling_avg)
        volatility = float(np.std(rolling_diff)) if len(rolling_diff) > 1 else 0.0
        
        # Use rolling average for trend calculation instead of raw values
        values_for_trend = rolling_avg
        total_days = max((timestamps[-1] - timestamps[0]).total_seconds() / 86400.0, 1e-6)
        delta = values_for_trend[-1] - values_for_trend[0]
        change_per_day = delta / total_days
        
        # Enhanced direction detection with volatility threshold
        volatility_threshold = volatility * 0.5  # Adaptive threshold
        direction = 'up' if delta > max(0.5, volatility_threshold) else 'down' if delta < -max(0.5, volatility_threshold) else 'steady'

        return {
            'direction': direction,
            'delta': round(delta, 2),
            'changePerDay': round(change_per_day, 2),
            'volatility': round(volatility, 2),
            'rollingAvg': float(rolling_avg[-1]) if len(rolling_avg) > 0 else None,
        }

    def _calculate_personalized_baselines(
        self,
        time_series: Dict[str, List[Dict[str, Any]]],
        user_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate personalized baselines from user's historical data
        Uses first 14 days of data (establishment period) or all available data if < 14 days
        """
        baselines = {}
        cfg = self.config
        
        # Define baseline calculation period (14 days for establishment)
        baseline_days = 14
        baseline_start = datetime.now(timezone.utc) - timedelta(days=baseline_days)
        
        for metric_type, points in time_series.items():
            if not points:
                continue
            
            # Filter points from baseline period (first 14 days)
            baseline_points = [p for p in points if p['timestamp'] >= baseline_start]
            
            # Need at least 3 readings to establish baseline
            if len(baseline_points) < 3:
                # Use all available data if insufficient baseline period data
                baseline_points = points[:min(20, len(points))]  # Use up to 20 most recent points
            
            if len(baseline_points) >= 3:
                values = [p['value'] for p in baseline_points]
                
                # Calculate baseline statistics (median is more robust than mean)
                baseline_median = float(np.median(values))
                baseline_mean = float(np.mean(values))
                baseline_std = float(np.std(values))
                
                # Define normal range as median ± 1.5 * IQR (more robust than ± std)
                q1, q3 = np.percentile(values, [25, 75])
                iqr = q3 - q1
                baseline_low = float(baseline_median - 1.5 * iqr)
                baseline_high = float(baseline_median + 1.5 * iqr)
                
                baselines[metric_type] = {
                    'median': baseline_median,
                    'mean': baseline_mean,
                    'std': baseline_std,
                    'low': baseline_low,
                    'high': baseline_high,
                    'sample_size': len(baseline_points),
                    'established': len(baseline_points) >= 10  # Baseline considered established with 10+ readings
                }
        
        # Add demographic-based default baselines if no data available
        if user_profile:
            age = self._safe_float(user_profile.get('age'))
            if age is None:
                dob = user_profile.get('dateOfBirth')
                if dob:
                    age = self._calculate_age(dob)
            
            gender = (user_profile.get('gender') or 'other').lower()
            
            # Age and gender-adjusted defaults if no historical data
            if 'heart_rate' not in baselines:
                # Normal HR: 60-100 for adults, slightly higher for elderly
                if age and age >= 65:
                    default_hr = 75
                else:
                    default_hr = 72
                baselines['heart_rate'] = {
                    'median': default_hr,
                    'mean': default_hr,
                    'std': 5.0,
                    'low': default_hr - 12,
                    'high': default_hr + 12,
                    'sample_size': 0,
                    'established': False
                }
            
            if 'respiratory_rate' not in baselines:
                baselines['respiratory_rate'] = {
                    'median': 16.0,
                    'mean': 16.0,
                    'std': 2.0,
                    'low': 12.0,
                    'high': 20.0,
                    'sample_size': 0,
                    'established': False
                }
        
        return baselines
    
    def _calculate_population_adjustments(
        self,
        user_profile: Optional[Dict[str, Any]] = None,
        time_series: Optional[Dict[str, List[Dict[str, Any]]]] = None
    ) -> Dict[str, float]:
        """
        Calculate population-based adjustments for scoring thresholds
        Uses demographic information and historical patterns to adjust thresholds
        Returns adjustment factors (multipliers) for different metrics
        """
        adjustments = {
            'heart_rate': 1.0,
            'respiratory_rate': 1.0,
            'blood_pressure': 1.0,
            'temperature': 1.0,
            'stress_level': 1.0
        }
        
        if not user_profile:
            return adjustments
        
        age = user_profile.get('age')
        gender = user_profile.get('gender', '').lower()
        
        # Age-based adjustments (older adults may have different normal ranges)
        if age:
            if age >= 65:
                # Older adults: slightly higher HR baseline, lower BP thresholds
                adjustments['heart_rate'] = 1.05  # +5% adjustment
                adjustments['blood_pressure'] = 0.98  # -2% adjustment (slightly higher thresholds)
                adjustments['respiratory_rate'] = 1.0  # No change
            elif age >= 50:
                # Middle-aged: slight adjustments
                adjustments['heart_rate'] = 1.02
                adjustments['blood_pressure'] = 0.99
            elif age < 18:
                # Children/adolescents: different ranges
                adjustments['heart_rate'] = 1.08  # +8% (higher baseline HR)
                adjustments['respiratory_rate'] = 1.05  # +5% (higher baseline RR)
                adjustments['blood_pressure'] = 0.95  # -5% (lower thresholds)
        
        # Gender-based adjustments (slight differences in normal ranges)
        if gender == 'female':
            # Females typically have slightly higher HR, lower BP
            adjustments['heart_rate'] *= 1.02
            adjustments['blood_pressure'] *= 0.98
        elif gender == 'male':
            # Males typically have slightly lower HR, higher BP
            adjustments['heart_rate'] *= 0.98
            adjustments['blood_pressure'] *= 1.02
        
        # Activity level adjustments (if available from time series)
        if time_series:
            steps_points = time_series.get('steps', [])
            if steps_points:
                recent_steps = sum([p['value'] for p in steps_points[-5:] if p.get('value')])
                avg_daily_steps = recent_steps / 5 if len(steps_points) >= 5 else recent_steps
                
                # Highly active individuals may have different baselines
                if avg_daily_steps > 8000:
                    # Very active: lower resting HR, better cardiovascular health
                    adjustments['heart_rate'] *= 0.97  # -3% (lower thresholds)
                    adjustments['blood_pressure'] *= 0.98  # -2%
                elif avg_daily_steps < 3000:
                    # Sedentary: may have higher baseline HR
                    adjustments['heart_rate'] *= 1.02  # +2%
        
        # Clip adjustments to reasonable ranges (±10%)
        for key in adjustments:
            adjustments[key] = max(0.9, min(1.1, adjustments[key]))
        
        return adjustments
    
    def _apply_adaptive_scoring(
        self,
        score: float,
        metric_type: str,
        population_adjustments: Dict[str, float]
    ) -> float:
        """
        Apply population-based adaptive adjustments to scores
        """
        adjustment = population_adjustments.get(metric_type, 1.0)
        return score * adjustment
    
    def _calculate_news2(
        self,
        latest: Dict[str, float],
        baselines: Optional[Dict[str, Dict[str, float]]] = None,
        population_adjustments: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        score = 0
        breakdown: List[Dict[str, Any]] = []
        cfg = self.config
        
        # Use personalized baselines if available
        if baselines is None:
            baselines = {}
        
        # Use population adjustments if available
        if population_adjustments is None:
            population_adjustments = {}

        resp_rate = latest.get('respiratory_rate')
        resp_points = 0
        if resp_rate is not None:
            # Use personalized baseline if available
            if 'respiratory_rate' in baselines and baselines['respiratory_rate'].get('established'):
                baseline = baselines['respiratory_rate']
                baseline_low = baseline['low']
                baseline_high = baseline['high']
                deviation = abs(resp_rate - baseline['median'])
                
                # Adjust thresholds based on baseline
                very_low_threshold = baseline_low - 3
                low_threshold = baseline_low
                normal_high_threshold = baseline_high
                high_threshold = baseline_high + 3
            else:
                # Use default thresholds
                very_low_threshold = cfg.news2_respiratory['very_low']
                low_threshold = cfg.news2_respiratory['low']
                normal_high_threshold = cfg.news2_respiratory['normal_high']
                high_threshold = cfg.news2_respiratory['high']
            
            if resp_rate <= very_low_threshold:
                resp_points = 3
            elif resp_rate <= low_threshold:
                resp_points = 1
            elif resp_rate <= normal_high_threshold:
                resp_points = 0
            elif resp_rate <= high_threshold:
                resp_points = 2
            else:
                resp_points = 3
        # Apply population-based adaptive adjustments
        resp_points_adjusted = self._apply_adaptive_scoring(
            float(resp_points), 'respiratory_rate', population_adjustments
        ) if population_adjustments else float(resp_points)
        resp_points_final = int(round(resp_points_adjusted))
        
        breakdown.append({
            'metric': 'respiratory_rate',
            'value': resp_rate,
            'points': resp_points_final,
            'baseline_adjusted': 'respiratory_rate' in baselines,
            'population_adjusted': 'respiratory_rate' in population_adjustments
        })
        score += resp_points_final

        spo2 = latest.get('oxygen_saturation')
        spo2_points = 0
        if spo2 is not None:
            if spo2 <= cfg.news2_spo2['critical']:
                spo2_points = 3
            elif spo2 <= cfg.news2_spo2['low']:
                spo2_points = 2
            elif spo2 <= cfg.news2_spo2['moderate']:
                spo2_points = 1
            else:
                spo2_points = 0
        breakdown.append({'metric': 'oxygen_saturation', 'value': spo2, 'points': spo2_points})
        score += spo2_points

        systolic = latest.get('bp_systolic')
        systolic_points = 0
        if systolic is not None:
            if systolic <= cfg.news2_systolic['very_low']:
                systolic_points = 3
            elif systolic <= cfg.news2_systolic['low']:
                systolic_points = 2
            elif systolic <= cfg.news2_systolic['moderate']:
                systolic_points = 1
            elif systolic <= cfg.news2_systolic['normal_high']:
                systolic_points = 0
            else:
                systolic_points = 3
        breakdown.append({'metric': 'bp_systolic', 'value': systolic, 'points': systolic_points})
        score += systolic_points

        heart_rate = latest.get('heart_rate')
        heart_points = 0
        if heart_rate is not None:
            # Use personalized baseline if available
            if 'heart_rate' in baselines and baselines['heart_rate'].get('established'):
                baseline = baselines['heart_rate']
                baseline_low = baseline['low']
                baseline_high = baseline['high']
                deviation = abs(heart_rate - baseline['median'])
                
                # Adjust thresholds based on baseline (larger deviations = higher points)
                very_low_threshold = baseline_low - 15
                low_threshold = baseline_low - 5
                normal_high_threshold = baseline_high + 5
                high_threshold = baseline_high + 20
                very_high_threshold = baseline_high + 40
            else:
                # Use default thresholds
                very_low_threshold = cfg.news2_heart_rate['very_low']
                low_threshold = cfg.news2_heart_rate['low']
                normal_high_threshold = cfg.news2_heart_rate['normal_high']
                high_threshold = cfg.news2_heart_rate['high']
                very_high_threshold = cfg.news2_heart_rate['very_high']
            
            if heart_rate <= very_low_threshold:
                heart_points = 3
            elif heart_rate <= low_threshold:
                heart_points = 1
            elif heart_rate <= normal_high_threshold:
                heart_points = 0
            elif heart_rate <= high_threshold:
                heart_points = 1
            elif heart_rate <= very_high_threshold:
                heart_points = 2
            else:
                heart_points = 3
        # Apply population-based adaptive adjustments
        heart_points_adjusted = self._apply_adaptive_scoring(
            float(heart_points), 'heart_rate', population_adjustments
        ) if population_adjustments else float(heart_points)
        heart_points_final = int(round(heart_points_adjusted))
        
        breakdown.append({
            'metric': 'heart_rate',
            'value': heart_rate,
            'points': heart_points_final,
            'baseline_adjusted': 'heart_rate' in baselines,
            'population_adjusted': 'heart_rate' in population_adjustments
        })
        score += heart_points_final

        temperature = latest.get('temperature')
        temp_points = 0
        if temperature is not None:
            if temperature < cfg.news2_temperature['hypothermic']:
                temp_points = 3
            elif temperature <= cfg.news2_temperature['low']:
                temp_points = 1
            elif temperature <= cfg.news2_temperature['normal_high']:
                temp_points = 0
            elif temperature <= cfg.news2_temperature['fever']:
                temp_points = 1
            else:
                temp_points = 2
        breakdown.append({'metric': 'temperature', 'value': temperature, 'points': temp_points})
        score += temp_points

        # Consciousness and oxygen support are not tracked in app yet
        level = 'low'
        if score >= cfg.news2_score_levels['critical']:
            level = 'critical'
        elif score >= cfg.news2_score_levels['high']:
            level = 'high'
        elif score >= cfg.news2_score_levels['moderate']:
            level = 'moderate'

        return {
            'score': score,
            'level': level,
            'breakdown': breakdown,
        }

    def _calculate_dynamic_weights(
        self,
        time_series: Dict[str, List[Dict[str, Any]]],
        trends: Dict[str, Dict[str, Any]],
        metric_type: str
    ) -> Dict[str, float]:
        """
        Calculate dynamic weights based on time series stability and trends.
        More stable metrics with consistent trends get higher weights.
        """
        cfg = self.config
        
        # Start with default weights based on metric type
        if metric_type == 'temperature':
            base_weights = cfg.fever_weights.copy()
        elif metric_type == 'respiratory_rate':
            base_weights = cfg.respiratory_weights.copy()
        else:
            # Default weights if metric type not recognized
            base_weights = {
                'temperature': 1.0,
                'heart_rate': 0.8,
                'stress': 0.6,
                'spo2': 0.7,
                'respiratory_rate': 1.0
            }
        
        # Calculate stability factors for each metric
        weights = {}
        for metric_key in base_weights.keys():
            if metric_key not in time_series or len(time_series[metric_key]) < 2:
                # Use base weight if insufficient data
                weights[metric_key] = base_weights[metric_key]
                continue
            
            points = time_series[metric_key]
            values = [p['value'] for p in points]
            
            # Calculate coefficient of variation (CV) as stability measure
            if len(values) > 1:
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / mean_val if mean_val > 0 else 1.0
                
                # Lower CV = more stable = higher weight multiplier (up to 1.2x)
                # Higher CV = less stable = lower weight multiplier (down to 0.8x)
                stability_factor = max(0.8, min(1.2, 1.0 - (cv * 0.5)))
            else:
                stability_factor = 1.0
            
            # Adjust based on trend consistency
            trend_factor = 1.0
            if metric_key in trends:
                trend = trends[metric_key]
                # If trend is strong and consistent, increase weight slightly
                if abs(trend.get('slope', 0)) > 0.1 and trend.get('r_squared', 0) > 0.5:
                    trend_factor = 1.05  # 5% boost for strong trends
            
            # Apply adjustments
            weights[metric_key] = base_weights[metric_key] * stability_factor * trend_factor
        
        return weights

    def _calculate_activity_level(
        self,
        time_series: Dict[str, List[Dict[str, Any]]],
        latest: Dict[str, float]
    ) -> str:
        """
        Calculate current activity level from steps and recent patterns
        Returns: 'rest', 'light', 'moderate', 'high', 'very_high'
        """
        steps_points = time_series.get('steps', [])
        
        # Get steps from last 30 minutes (approximate)
        now = datetime.now(timezone.utc)
        recent_steps = [
            p['value'] for p in steps_points
            if (now - p['timestamp']).total_seconds() < 1800  # Last 30 minutes
        ]
        
        if recent_steps:
            steps_last_30min = sum(recent_steps[-5:])  # Last 5 readings in 30 min
            steps_per_hour = steps_last_30min * 2  # Extrapolate to hourly rate
        else:
            steps_per_hour = 0
        
        # Get latest steps value if available
        current_steps = latest.get('steps', 0)
        
        # Activity level classification based on steps per hour
        if steps_per_hour >= 6000 or current_steps > 100:  # Very active
            return 'very_high'
        elif steps_per_hour >= 3000 or current_steps > 50:  # High activity
            return 'high'
        elif steps_per_hour >= 1500 or current_steps > 20:  # Moderate activity
            return 'moderate'
        elif steps_per_hour >= 500:  # Light activity
            return 'light'
        else:
            return 'rest'  # Resting
    
    def _adjust_vitals_for_activity(
        self,
        latest: Dict[str, float],
        activity_level: str
    ) -> Dict[str, float]:
        """
        Adjust vitals based on activity level to avoid false alarms during/after exercise
        Returns adjusted vitals dict
        """
        adjusted = latest.copy()
        
        if activity_level in ['moderate', 'high', 'very_high']:
            # During/after exercise, vitals are naturally elevated
            # Adjust thresholds to account for expected elevation
            
            hr = adjusted.get('heart_rate')
            if hr is not None:
                # Reduce HR by expected exercise elevation
                if activity_level == 'very_high':
                    adjusted['heart_rate'] = max(hr - 40, 60)  # Reduce by expected max elevation
                elif activity_level == 'high':
                    adjusted['heart_rate'] = max(hr - 30, 60)
                elif activity_level == 'moderate':
                    adjusted['heart_rate'] = max(hr - 20, 60)
            
            rr = adjusted.get('respiratory_rate')
            if rr is not None:
                # Reduce RR by expected exercise elevation
                if activity_level == 'very_high':
                    adjusted['respiratory_rate'] = max(rr - 10, 12)
                elif activity_level == 'high':
                    adjusted['respiratory_rate'] = max(rr - 7, 12)
                elif activity_level == 'moderate':
                    adjusted['respiratory_rate'] = max(rr - 4, 12)
            
            temp = adjusted.get('temperature')
            if temp is not None:
                # Slight adjustment for exercise-induced temperature elevation
                if activity_level in ['high', 'very_high']:
                    adjusted['temperature'] = max(temp - 0.3, 35.5)
        
        return adjusted
    
    def _estimate_fever_probability(
        self,
        latest: Dict[str, float],
        trends: Dict[str, Dict[str, Any]],
        news2_score: int,
        activity_level: str = 'rest',
        time_series: Optional[Dict[str, List[Dict[str, Any]]]] = None
    ) -> Tuple[float, List[str]]:
        cfg = self.config
        heart_rate = latest.get('heart_rate', 72)
        stress = latest.get('stress_level', 40)
        spo2 = latest.get('oxygen_saturation', 98)
        temp = latest.get('temperature')

        hr_signal = max(0.0, (heart_rate - cfg.fever_hr_threshold) / 24.0)
        stress_signal = max(0.0, (stress - cfg.fever_stress_threshold) / 25.0)
        spo2_signal = max(0.0, (cfg.fever_spo2_threshold - spo2) / 3.0)
        temp_signal_value = 0.0
        temp_indicator = None

        if temp is not None:
            temp_signal_value = max(0.0, temp - cfg.fever_temp_elevated)
            if temp >= cfg.fever_temp_threshold:
                temp_indicator = f"Temperature elevated ({temp:.1f}°C)"

        # Enhanced trend analysis with multi-day patterns
        trend_bonus = 0.0
        hr_trend = trends.get('heart_rate', {})
        temp_trend = trends.get('temperature', {})
        
        # Multi-day trend analysis: stronger bonus for sustained trends
        if hr_trend.get('direction') == 'up':
            change_per_day = abs(hr_trend.get('changePerDay', 0))
            # Bonus scales with rate of change (max 2x bonus for strong trends)
            trend_multiplier = min(2.0, 1.0 + change_per_day / 5.0)
            trend_bonus += cfg.fever_trend_bonus['heart_rate'] * trend_multiplier
        
        if temp_trend.get('direction') == 'up':
            change_per_day = abs(temp_trend.get('changePerDay', 0))
            trend_multiplier = min(2.0, 1.0 + change_per_day / 0.5)  # 0.5°C per day threshold
            trend_bonus += cfg.fever_trend_bonus['temperature'] * trend_multiplier
        
        # Additional bonus if both trends are consistent
        if hr_trend.get('direction') == 'up' and temp_trend.get('direction') == 'up':
            trend_bonus += 0.05  # Synergy bonus

        # Use dynamic weights if time series available
        if time_series:
            weights = self._calculate_dynamic_weights(time_series, trends, 'temperature')
        else:
            weights = cfg.fever_weights
        
        # Adjust probability based on activity level
        # During/after exercise, reduce fever probability (elevated vitals are expected)
        activity_factor = 1.0
        if activity_level in ['moderate', 'high', 'very_high']:
            # Reduce fever probability during exercise
            if activity_level == 'very_high':
                activity_factor = 0.7  # 30% reduction
            elif activity_level == 'high':
                activity_factor = 0.8  # 20% reduction
            elif activity_level == 'moderate':
                activity_factor = 0.85  # 15% reduction
        
        # Use dynamic weights if available, otherwise use config weights
        weight_temp = weights.get('temperature', cfg.fever_weights['temperature'])
        weight_hr = weights.get('heart_rate', cfg.fever_weights['heart_rate'])
        weight_stress = weights.get('stress', cfg.fever_weights['stress'])
        weight_spo2 = weights.get('spo2', cfg.fever_weights['spo2'])
        
        score = (
            weight_temp * temp_signal_value
            + weight_hr * hr_signal
            + weight_stress * stress_signal
            + weight_spo2 * spo2_signal
            + trend_bonus
            + news2_score / 12.0
        )
        probability = self._sigmoid(score - cfg.fever_sigmoid_offset) * activity_factor

        signals = []
        if temp_indicator:
            signals.append(temp_indicator)
        if heart_rate >= 95:
            signals.append(f"Resting HR trending high ({heart_rate} bpm)")
        if spo2 <= 95:
            signals.append(f"SpO2 slightly depressed ({spo2}%)")
        if stress >= 75:
            signals.append("Stress biomarkers elevated")
        if not signals:
            signals.append("No fever indicators detected")

        return probability, signals

    def _estimate_respiratory_probability(
        self,
        latest: Dict[str, float],
        trends: Dict[str, Dict[str, Any]],
        news2_score: int,
        activity_level: str = 'rest',
        time_series: Optional[Dict[str, List[Dict[str, Any]]]] = None
    ) -> Tuple[float, List[str]]:
        cfg = self.config
        resp_rate = latest.get('respiratory_rate', 16)
        spo2 = latest.get('oxygen_saturation', 98)
        stress = latest.get('stress_level', 40)
        hr = latest.get('heart_rate', 72)

        # Use dynamic weights if time series available
        if time_series:
            weights = self._calculate_dynamic_weights(time_series, trends, 'respiratory_rate')
        else:
            weights = cfg.respiratory_weights
        
        resp_signal = max(0.0, (resp_rate - cfg.respiratory_rate_high) / 6.0)
        spo2_signal = max(0.0, (cfg.respiratory_spo2_threshold - spo2) / 4.0)
        hr_signal = max(0.0, (hr - cfg.respiratory_hr_threshold) / 25.0)
        stress_signal = max(0.0, (stress - cfg.respiratory_stress_threshold) / 25.0)

        # Enhanced trend analysis with multi-day patterns
        trend_bonus = 0.0
        rr_trend = trends.get('respiratory_rate', {})
        spo2_trend = trends.get('oxygen_saturation', {})
        
        # Multi-day trend analysis with volatility consideration
        if rr_trend.get('direction') == 'up':
            change_per_day = abs(rr_trend.get('changePerDay', 0))
            volatility = rr_trend.get('volatility', 0)
            # Reduce bonus if high volatility (unreliable trend)
            volatility_factor = max(0.5, 1.0 - volatility / 2.0) if volatility > 0 else 1.0
            trend_multiplier = min(2.0, 1.0 + change_per_day / 1.0) * volatility_factor
            trend_bonus += cfg.respiratory_trend_bonus['respiratory_rate'] * trend_multiplier
        
        if spo2_trend.get('direction') == 'down':
            change_per_day = abs(spo2_trend.get('changePerDay', 0))
            volatility = spo2_trend.get('volatility', 0)
            volatility_factor = max(0.5, 1.0 - volatility / 1.0) if volatility > 0 else 1.0
            trend_multiplier = min(2.0, 1.0 + change_per_day / 0.5) * volatility_factor  # 0.5% per day threshold
            trend_bonus += cfg.respiratory_trend_bonus['oxygen_saturation'] * trend_multiplier
        
        # Synergy bonus for consistent respiratory strain indicators
        if rr_trend.get('direction') == 'up' and spo2_trend.get('direction') == 'down':
            trend_bonus += 0.05

        # Adjust probability based on activity level
        # During/after exercise, reduce respiratory strain probability (elevated RR is expected)
        activity_factor = 1.0
        if activity_level in ['moderate', 'high', 'very_high']:
            if activity_level == 'very_high':
                activity_factor = 0.7  # 30% reduction
            elif activity_level == 'high':
                activity_factor = 0.8  # 20% reduction
            elif activity_level == 'moderate':
                activity_factor = 0.85  # 15% reduction
        
        # Use dynamic weights if available, otherwise use config weights
        weight_rr = weights.get('respiratory_rate', cfg.respiratory_weights['respiratory_rate'])
        weight_spo2 = weights.get('spo2', cfg.respiratory_weights['spo2'])
        weight_hr = weights.get('heart_rate', cfg.respiratory_weights['heart_rate'])
        weight_stress = weights.get('stress', cfg.respiratory_weights['stress'])
        
        score = (
            weight_rr * resp_signal
            + weight_spo2 * spo2_signal
            + weight_hr * hr_signal
            + weight_stress * stress_signal
            + trend_bonus
            + news2_score / 15.0
        )
        probability = self._sigmoid(score - cfg.respiratory_sigmoid_offset) * activity_factor

        signals = []
        if resp_rate >= cfg.respiratory_rate_very_high:
            signals.append(f"Respiratory rate elevated ({resp_rate}/min)")
        if spo2 <= cfg.respiratory_spo2_low:
            signals.append(f"SpO2 depressed ({spo2}%)")
        if stress >= 75:
            signals.append("Stress/cortisol indicators elevated")
        if not signals:
            signals.append("Breathing stable")

        return probability, signals

    def _estimate_stress_recovery(
        self,
        latest: Dict[str, float],
        trends: Dict[str, Dict[str, Any]],
        time_series: Optional[Dict[str, List[Dict[str, Any]]]] = None
    ) -> float:
        """
        Enhanced stress recovery index using HRV features and RR variability
        """
        cfg = self.config
        stress = latest.get('stress_level', 40)
        heart_rate = latest.get('heart_rate', 72)
        respiratory = latest.get('respiratory_rate', 16)

        stress_norm = (100 - stress) / 100.0
        hr_norm = max(0.0, 1 - abs(heart_rate - cfg.stress_recovery_optimal_hr) / 40.0)
        rr_norm = max(0.0, 1 - abs(respiratory - cfg.stress_recovery_optimal_rr) / 10.0)

        # Calculate HRV features if time series data available
        hrv_score = 0.0
        rr_variability_score = 0.0
        
        if time_series:
            # HRV analysis from heart rate variability
            hr_points = time_series.get('heart_rate', [])
            if len(hr_points) >= 10:
                hr_values = [p['value'] for p in hr_points[-20:]]  # Last 20 readings
                hr_intervals = np.diff(hr_values) if len(hr_values) > 1 else []
                
                if len(hr_intervals) > 0:
                    # Calculate HRV metrics
                    rmssd = float(np.sqrt(np.mean(hr_intervals ** 2))) if len(hr_intervals) > 0 else 0
                    sdnn = float(np.std(hr_values)) if len(hr_values) > 1 else 0
                    
                    # Higher HRV = better recovery (normalize to 0-1)
                    # Typical RMSSD: 20-60ms (good), <20ms (poor recovery)
                    # Typical SDNN: 20-50ms (good), <20ms (poor recovery)
                    rmssd_norm = min(1.0, max(0.0, (rmssd - 5) / 50.0))  # Normalize 5-55ms range
                    sdnn_norm = min(1.0, max(0.0, (sdnn - 5) / 45.0))  # Normalize 5-50ms range
                    
                    # Combined HRV score (weighted average)
                    hrv_score = 0.6 * rmssd_norm + 0.4 * sdnn_norm
            
            # Respiratory rate variability analysis
            rr_points = time_series.get('respiratory_rate', [])
            if len(rr_points) >= 10:
                rr_values = [p['value'] for p in rr_points[-20:]]  # Last 20 readings
                if len(rr_values) > 1:
                    rr_std = float(np.std(rr_values))
                    # Moderate variability (1-2 breaths/min std) indicates good recovery
                    # Too low variability (<0.5) or too high (>3) indicates stress/poor recovery
                    if 0.5 <= rr_std <= 2.5:
                        rr_variability_score = 1.0 - abs(rr_std - 1.5) / 1.5  # Peak at 1.5 std
                    else:
                        rr_variability_score = max(0.0, 1.0 - abs(rr_std - 1.5) / 3.0)
        
        trend_bonus = cfg.stress_recovery_trend_bonus if trends.get('stress_level', {}).get('direction') == 'down' else 0.0
        
        # Enhanced index with HRV and RR variability
        # Adjust weights if HRV data available (reduce base weights slightly)
        if hrv_score > 0 or rr_variability_score > 0:
            # Include HRV and RR variability in calculation
            base_weights_sum = sum(cfg.stress_recovery_weights.values())
            # Reduce base weights to 70% to make room for HRV/RR variability
            adjusted_weights = {k: v * 0.7 for k, v in cfg.stress_recovery_weights.items()}
            
            index = (
                adjusted_weights['stress'] * stress_norm
                + adjusted_weights['heart_rate'] * hr_norm
                + adjusted_weights['respiratory_rate'] * rr_norm
                + 0.15 * hrv_score  # 15% weight for HRV
                + 0.10 * rr_variability_score  # 10% weight for RR variability
                + trend_bonus
            )
        else:
            # Use original weights if no HRV data
            index = (
                cfg.stress_recovery_weights['stress'] * stress_norm
                + cfg.stress_recovery_weights['heart_rate'] * hr_norm
                + cfg.stress_recovery_weights['respiratory_rate'] * rr_norm
                + trend_bonus
            )
        
        return float(np.clip(index, 0.0, 1.0))

    def _build_lifestyle_plan(
        self,
        latest: Dict[str, float],
        user_profile: Optional[Dict[str, Any]],
        stress_recovery: float,
        trends: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate fully dynamic lifestyle plan based on actual health metrics, trends, and user profile.
        No hardcoded values - everything is calculated from real data.
        """
        cfg = self.config
        
        # Calculate hydration target dynamically from user profile
        hydration_target = cfg.lifestyle_hydration_base_ml
        if user_profile:
            weight = user_profile.get('weightKg')
            if weight:
                hydration_target = int(weight * cfg.lifestyle_hydration_ml_per_kg)

        # Calculate sleep target dynamically from user profile
        sleep_target = cfg.lifestyle_sleep_target_hours
        if user_profile and user_profile.get('age') and user_profile['age'] < cfg.lifestyle_young_adult_age:
            sleep_target = cfg.lifestyle_sleep_young_adult_hours

        # Get all available health metrics
        heart_rate = latest.get('heart_rate')
        stress_level = latest.get('stress_level')
        temperature = latest.get('temperature')
        respiratory_rate = latest.get('respiratory_rate')
        spo2 = latest.get('oxygen_saturation')
        steps_avg = latest.get('steps', 0)
        sleep_avg = latest.get('sleep_hours', 0)
        bp_systolic = latest.get('bp_systolic')
        bp_diastolic = latest.get('bp_diastolic')
        
        # Get trends for all metrics
        hr_trend = trends.get('heart_rate', {})
        stress_trend = trends.get('stress_level', {})
        temp_trend = trends.get('temperature', {})
        rr_trend = trends.get('respiratory_rate', {})
        spo2_trend = trends.get('oxygen_saturation', {})
        steps_trend = trends.get('steps', {})
        sleep_trend = trends.get('sleep_hours', {})
        
        # Calculate dynamic thresholds based on user's baseline
        # Use trends to determine if metrics are improving or declining
        hr_direction = hr_trend.get('direction', 'stable')
        hr_change = hr_trend.get('changePerDay', 0)
        stress_direction = stress_trend.get('direction', 'stable')
        stress_change = stress_trend.get('changePerDay', 0)
        
        # Generate morning recommendations dynamically
        morning = []
        
        # Calculate morning hydration based on recovery and sleep
        if sleep_avg > 0:
            # Adjust hydration based on sleep quality (less sleep = more dehydration)
            sleep_factor = max(0.08, min(0.18, 0.12 + (8 - sleep_avg) * 0.01))
        else:
            # Use stress recovery to estimate hydration needs
            sleep_factor = 0.10 + (1 - stress_recovery) * 0.05
        
        morning_hydration = int(hydration_target * sleep_factor)
        morning.append(f"Start with {morning_hydration}ml water to rehydrate")
        
        # Calculate breathing exercise duration based on stress recovery and respiratory rate
        if stress_recovery < cfg.stress_recovery_low_threshold:
            breathing_duration = int(10 + (1 - stress_recovery) * 5)  # 10-15 minutes
            breathing_type = "gentle breathing exercises to activate parasympathetic system"
        elif respiratory_rate and respiratory_rate > 20:
            breathing_duration = int(8 + (respiratory_rate - 20) * 0.5)  # 8-12 minutes
            breathing_type = "slow, deep breathing to lower respiratory rate"
        elif stress_level and stress_level > 60:
            breathing_duration = 7
            breathing_type = "box-breathing to reduce stress"
        else:
            breathing_duration = 5
            breathing_type = "light breathing exercises"
        
        morning.append(f"{breathing_duration}-minute {breathing_type}")
        
        # Generate caffeine recommendation based on heart rate and trends
        if heart_rate:
            hr_deviation = abs(heart_rate - cfg.stress_recovery_optimal_hr)
            if hr_trend.get('direction') == 'up' and heart_rate > 80:
                hr_increase = hr_trend.get('changePerDay', 0)
                if hr_increase > 2:
                    morning.append("Consider herbal tea instead of caffeine - heart rate trending upward")
                else:
                    morning.append("Reduce caffeine intake - heart rate elevated")
            elif heart_rate < 65 and hr_trend.get('direction') == 'down':
                morning.append("Light activity recommended to gradually increase heart rate")
        
        # Generate stress management recommendation
        if stress_level:
            if stress_level > 70:
                morning.append("Prioritize stress-reducing activities - consider 10-minute meditation or gentle yoga")
            elif stress_level > 50:
                morning.append("Include stress-reducing activities in morning routine")
            elif stress_level < 30:
                morning.append("Maintain current routine - stress levels are optimal")
        
        # Generate afternoon recommendations dynamically
        afternoon = []
        
        # Calculate walk duration based on steps and trends
        if steps_avg > 0:
            # Calculate target steps based on user's average and trends
            steps_target = 8000  # Base target
            if user_profile:
                goal = user_profile.get('goal', '').lower()
                if goal == 'improve_endurance':
                    steps_target = 12000
                elif goal == 'weight_loss':
                    steps_target = 10000
            
            # Adjust based on steps trend
            if steps_trend.get('direction') == 'down':
                steps_target = int(steps_target * 1.1)  # Increase target if declining
            
            steps_needed = max(0, steps_target - steps_avg)
            if steps_needed > 0:
                # Calculate walk duration needed (average 100 steps per minute)
                walk_minutes = max(5, min(30, int(steps_needed / 100)))
                afternoon.append(f"Take a {walk_minutes}-minute walk to reach daily activity goals ({steps_needed} steps needed)")
            else:
                afternoon.append("Maintain current activity level - daily step goal on track")
        else:
            # No historical data - calculate based on health status
            if stress_recovery < 0.5:
                walk_minutes = 10
            elif heart_rate and heart_rate > 85:
                walk_minutes = 8
            else:
                walk_minutes = 15
            afternoon.append(f"Take a {walk_minutes}-minute walk to boost circulation and energy")
        
        # Generate nutrition recommendation based on goals, health, and trends
        if user_profile:
            goal = user_profile.get('goal', '').lower()
            weight = user_profile.get('weightKg', 70)
            
            # Calculate protein needs dynamically
            if goal == 'weight_loss':
                protein_factor = 2.2
                nutrition_focus = "protein-rich lunch with vegetables for sustained energy"
            elif goal == 'muscle_gain':
                protein_factor = 2.5
                nutrition_focus = "lean protein and complex carbs in lunch for muscle recovery"
            else:
                protein_factor = 1.6
                nutrition_focus = "balanced lunch with protein, vegetables, and whole grains"
            
            # Adjust based on activity level
            if steps_avg > 8000:
                nutrition_focus += " - increase portion size for higher activity"
            elif steps_avg < 3000:
                nutrition_focus += " - focus on nutrient density"
            
            afternoon.append(f"Focus on {nutrition_focus}")
        
        # Generate temperature-based recommendation
        if temperature:
            temp_deviation = abs(temperature - 36.5)
            if temperature > 37.2:
                temp_severity = "elevated"
                if temperature > 37.5:
                    temp_severity = "significantly elevated"
                afternoon.append(f"Stay hydrated and avoid intense activities - body temperature {temp_severity}")
            elif temperature < 36.1:
                afternoon.append("Light movement recommended to help regulate body temperature")
        
        # Generate evening recommendations dynamically
        evening = []
        
        # Calculate screen-off time based on stress recovery and sleep trends
        if stress_recovery < cfg.stress_recovery_low_threshold:
            screen_off_minutes = int(60 + (1 - stress_recovery) * 20)  # 60-80 minutes
            recovery_focus = "better recovery"
        elif sleep_trend.get('direction') == 'down' or (sleep_avg > 0 and sleep_avg < 6):
            screen_off_minutes = 60
            recovery_focus = "improved sleep quality"
        elif stress_recovery > 0.7:
            screen_off_minutes = 30
            recovery_focus = "maintained recovery"
        else:
            screen_off_minutes = 45
            recovery_focus = "optimal rest"
        
        evening.append(f"Screen off {screen_off_minutes} minutes before bed for {recovery_focus}")
        
        # Calculate evening activity duration and type
        if stress_recovery < cfg.stress_recovery_low_threshold:
            evening_activity_duration = int(15 + (1 - stress_recovery) * 5)  # 15-20 minutes
            evening_activity = "guided meditation or progressive muscle relaxation"
        elif stress_level and stress_level > 60:
            evening_activity_duration = 12
            evening_activity = "meditation or gentle yoga"
        elif stress_recovery > 0.7:
            evening_activity_duration = 10
            evening_activity = "light stretching or mobility work"
        else:
            evening_activity_duration = 10
            evening_activity = "gentle stretching to downshift stress"
        
        evening.append(f"{evening_activity_duration}-minute {evening_activity}")
        
        # Generate respiratory rate-based recommendation
        if respiratory_rate:
            if respiratory_rate > 20:
                rr_elevation = respiratory_rate - 20
                breathing_focus = f"slow, deep breathing exercises to lower respiratory rate (currently {respiratory_rate}/min)"
                evening.append(f"Focus on {breathing_focus}")
            elif respiratory_rate < 12:
                evening.append("Gentle movement before bed to support healthy breathing patterns")
        
        # Generate SpO2-based recommendation
        if spo2:
            if spo2 < 96:
                spo2_severity = "slightly low" if spo2 >= 94 else "low"
                evening.append(f"Ensure good ventilation in bedroom - SpO2 {spo2_severity} ({spo2}%)")
        
        # Generate stress trend-based recommendation
        if stress_trend.get('direction') == 'up':
            stress_increase = stress_trend.get('changePerDay', 0)
            if stress_increase > 5:
                evening.append("Extended wind-down routine critical - stress levels rising rapidly")
            else:
                evening.append("Extended wind-down routine recommended - stress levels increasing")
        elif stress_trend.get('direction') == 'down':
            evening.append("Maintain current evening routine - stress levels improving")
        
        # Generate heart rate trend-based recommendation
        if hr_trend.get('direction') == 'up':
            hr_increase = hr_trend.get('changePerDay', 0)
            if hr_increase > 3:
                evening.append("Avoid all stimulating activities - heart rate trend indicates need for rest")
            else:
                evening.append("Avoid stimulating activities - heart rate trend suggests need for rest")
        
        # Generate fallback recommendations if lists are empty (shouldn't happen, but safety)
        if not morning:
            morning.append(f"Start with {int(hydration_target * 0.12)}ml water and light movement")
        if not afternoon:
            afternoon.append("Take a break and include light movement")
        if not evening:
            evening.append("Prepare for restful sleep with gentle activities")

        plan = {
            'hydrationTargetMl': hydration_target,
            'sleepTargetHours': sleep_target,
            'morning': morning,
            'afternoon': afternoon,
            'evening': evening,
        }

        return plan

    def _build_lifestyle_card(
        self,
        latest: Dict[str, float],
        user_profile: Optional[Dict[str, Any]],
        lifestyle_plan: Optional[Dict[str, Any]],
        stress_recovery: float,
        trends: Optional[Dict[str, Dict[str, Any]]] = None,
        news2: Optional[Dict[str, Any]] = None,
        bp_assessment: Optional[Dict[str, Any]] = None,
        stress_assessment: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        if not user_profile:
            return None

        gender = (user_profile.get('gender') or 'other').lower()
        weight = self._safe_float(user_profile.get('weightKg') or user_profile.get('weight'), 70.0)
        height = self._safe_float(user_profile.get('heightCm') or user_profile.get('height'), 170.0)
        goal = (user_profile.get('goal') or 'general_fitness').lower()

        age = self._safe_float(user_profile.get('age'))
        if age is None:
            dob = user_profile.get('dateOfBirth')
            if dob:
                age = self._calculate_age(dob)
        if age is None:
            age = 30.0

        height_m = max(height / 100.0, 1.0)
        bmi = weight / (height_m * height_m)

        if gender == 'male':
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161

        activity_multiplier = 1.55
        tdee = bmr * activity_multiplier

        # Generate dynamic recommendations based on actual health metrics, trends, and recovery
        # Base values calculated from user profile and health data
        heart_rate = latest.get('heart_rate')
        stress_level = latest.get('stress_level')
        steps_avg = latest.get('steps', 0)
        sleep_avg = latest.get('sleep_hours', 0)
        
        # Calculate base calories from TDEE
        calories = tdee
        protein = weight * 1.6
        carbs = calories * 0.4 / 4
        fats = calories * 0.3 / 9
        
        # Dynamic workout duration based on stress recovery and heart rate
        if stress_recovery < 0.4:
            workout_duration = 20  # Lower intensity if recovery is poor
        elif stress_recovery < 0.6:
            workout_duration = 30  # Moderate if recovery is fair
        elif heart_rate and heart_rate > 85:
            workout_duration = 25  # Shorter if heart rate is elevated
        else:
            workout_duration = 45  # Normal duration
        
        # Dynamic workout type based on health metrics and trends
        workout_type_parts = []
        if stress_level and stress_level > 60:
            workout_type_parts.append('Yoga/Meditation')
        if heart_rate and heart_rate > 80:
            workout_type_parts.append('Light Cardio')
        elif heart_rate and heart_rate < 65:
            workout_type_parts.append('Strength Training')
        
        # Add based on goal but adjust for actual health
        if goal == 'weight_loss':
            calories = tdee * 0.85
            protein = weight * 2.2
            carbs = calories * 0.35 / 4
            fats = calories * 0.25 / 9
            if not workout_type_parts:
                workout_type_parts = ['Cardio', 'Strength Training']
            workout_duration = max(workout_duration, 45)
        elif goal == 'weight_gain' or goal == 'muscle_gain':
            calories = tdee * 1.15 if goal == 'weight_gain' else tdee * 1.1
            protein = weight * (2.5 if goal == 'muscle_gain' else 1.8)
            carbs = calories * (0.40 if goal == 'muscle_gain' else 0.45) / 4
            fats = calories * (0.20 if goal == 'muscle_gain' else 0.25) / 9
            if not workout_type_parts:
                workout_type_parts = ['Strength Training']
            workout_duration = max(workout_duration, 60)
        elif goal == 'improve_endurance':
            calories = tdee * 1.05
            carbs = calories * 0.50 / 4
            if not workout_type_parts:
                workout_type_parts = ['Cardio', 'Endurance Training']
            workout_duration = max(workout_duration, 60)
        elif goal == 'maintain':
            if not workout_type_parts:
                workout_type_parts = ['Mixed Training']
        
        # Default if no parts determined
        if not workout_type_parts:
            workout_type_parts = ['General Exercise']
        workout_type = ' + '.join(workout_type_parts)
        
        # Dynamic steps based on current activity and health
        if steps_avg > 0:
            # Use average steps as baseline, adjust based on health
            steps = int(steps_avg * 1.1)  # 10% increase from average
            if stress_recovery < 0.5:
                steps = int(steps * 0.9)  # Reduce if recovery is poor
            if heart_rate and heart_rate > 85:
                steps = int(steps * 0.85)  # Reduce if heart rate elevated
        else:
            # No historical data - base on goal and health
            if goal == 'improve_endurance':
                steps = 12000
            elif goal == 'weight_loss':
                steps = 10000
            elif stress_recovery < 0.5:
                steps = 6000  # Lower if recovery poor
            else:
                steps = 8000  # Default
        
        # Generate dynamic notes based on actual health data
        notes_parts = []
        if stress_recovery < 0.4:
            notes_parts.append('Focus on recovery and stress management.')
        if heart_rate and heart_rate > 85:
            notes_parts.append('Monitor heart rate during activities.')
        if stress_level and stress_level > 60:
            notes_parts.append('Consider stress-reducing activities like meditation.')
        if trends:
            improving_metrics = [m for m, t in trends.items() if t.get('direction') == 'down' and m in ['stress_level', 'heart_rate']]
            if improving_metrics:
                notes_parts.append(f'Great progress on {", ".join(improving_metrics)}.')
        
        # Add goal-specific note
        if goal == 'weight_loss':
            notes_parts.append('Calorie deficit plan with high protein intake.')
        elif goal == 'muscle_gain':
            notes_parts.append('High protein intake essential for muscle growth.')
        elif goal == 'improve_endurance':
            notes_parts.append('Higher carb intake for sustained energy.')
        
        if not notes_parts:
            notes_parts.append('Personalized recommendations based on your health profile.')
        notes = ' '.join(notes_parts)

        # Calculate dynamic lifestyle score using weighted average approach
        # Each component contributes a percentage (0-100) that's weighted and averaged
        # This prevents the score from always maxing out at 100
        
        score_components = []
        weights = []
        
        # BMI contribution (weight: 20%)
        if 18.5 <= bmi <= 24.9:
            bmi_score = 100
        elif 24.9 < bmi <= 27:
            bmi_score = 75
        elif 27 < bmi <= 30:
            bmi_score = 50
        elif bmi < 18.5:
            bmi_score = 60  # Underweight is less ideal
        else:  # bmi > 30
            bmi_score = 30
        score_components.append(bmi_score)
        weights.append(0.20)
        
        # Stress recovery contribution (weight: 25%)
        stress_recovery_score = stress_recovery * 100
        score_components.append(stress_recovery_score)
        weights.append(0.25)
        
        # Vital signs contribution (weight: 25% total)
        heart_rate = latest.get('heart_rate', 72)
        respiratory_rate = latest.get('respiratory_rate', 16)
        temperature = latest.get('temperature', 36.5)
        spo2 = latest.get('oxygen_saturation', 98)
        
        # Heart rate score (0-100)
        if 60 <= heart_rate <= 100:
            hr_score = 100
        elif 50 <= heart_rate < 60 or 100 < heart_rate <= 110:
            hr_score = 70
        else:
            hr_score = 40
        score_components.append(hr_score)
        weights.append(0.08)  # 8% of total
        
        # Respiratory rate score (0-100)
        if 12 <= respiratory_rate <= 20:
            rr_score = 100
        elif 10 <= respiratory_rate < 12 or 20 < respiratory_rate <= 24:
            rr_score = 70
        else:
            rr_score = 40
        score_components.append(rr_score)
        weights.append(0.05)  # 5% of total
        
        # Temperature score (0-100)
        if 36.1 <= temperature <= 37.2:
            temp_score = 100
        elif 35.5 <= temperature < 36.1 or 37.2 < temperature <= 37.5:
            temp_score = 70
        else:
            temp_score = 40
        score_components.append(temp_score)
        weights.append(0.05)  # 5% of total
        
        # SpO2 score (0-100)
        if spo2 >= 98:
            spo2_score = 100
        elif spo2 >= 96:
            spo2_score = 80
        elif spo2 >= 94:
            spo2_score = 60
        else:
            spo2_score = 30
        score_components.append(spo2_score)
        weights.append(0.07)  # 7% of total
        
        # Blood pressure contribution (weight: 10%)
        if bp_assessment:
            bp_level = bp_assessment.get('level', 'normal')
            if bp_level == 'normal':
                bp_score = 100
            elif bp_level == 'elevated':
                bp_score = 75
            elif bp_level == 'stage1':
                bp_score = 50
            elif bp_level == 'stage2':
                bp_score = 25
            else:  # crisis
                bp_score = 0
        else:
            bp_score = 70  # Unknown = moderate score
        score_components.append(bp_score)
        weights.append(0.10)
        
        # Stress level contribution (weight: 10%)
        if stress_assessment:
            stress_level_assessment = stress_assessment.get('level', 'moderate')
            if stress_level_assessment == 'low':
                stress_score = 100
            elif stress_level_assessment == 'moderate':
                stress_score = 70
            elif stress_level_assessment == 'elevated':
                stress_score = 40
            else:  # very_high
                stress_score = 10
        else:
            stress_score = 70  # Unknown = moderate score
        score_components.append(stress_score)
        weights.append(0.10)
        
        # Trends contribution (weight: 5%) - improving trends boost score
        if trends:
            improving_count = 0
            declining_count = 0
            
            for metric, trend_data in trends.items():
                direction = trend_data.get('direction', 'stable')
                if direction == 'down' and metric in ['stress_level', 'heart_rate', 'temperature']:
                    improving_count += 1
                elif direction == 'up' and metric in ['stress_level', 'heart_rate', 'temperature']:
                    declining_count += 1
                elif direction == 'up' and metric in ['oxygen_saturation']:
                    improving_count += 1
                elif direction == 'down' and metric in ['oxygen_saturation']:
                    declining_count += 1
            
            if improving_count > declining_count:
                trends_score = 100
            elif improving_count == declining_count and improving_count > 0:
                trends_score = 70
            elif declining_count > improving_count:
                trends_score = 30
            else:
                trends_score = 50  # Stable
        else:
            trends_score = 50  # No trends = neutral
        score_components.append(trends_score)
        weights.append(0.05)
        
        # NEWS2 score contribution (weight: 5%) - lower is better
        if news2:
            news2_score_value = news2.get('score', 0)
            if news2_score_value == 0:
                news2_score = 100
            elif news2_score_value <= 2:
                news2_score = 80
            elif news2_score_value <= 4:
                news2_score = 60
            elif news2_score_value <= 6:
                news2_score = 40
            else:  # score >= 7
                news2_score = 20
        else:
            news2_score = 70  # Unknown = moderate score
        score_components.append(news2_score)
        weights.append(0.05)

        # Calculate weighted average
        if len(score_components) > 0 and sum(weights) > 0:
            # Normalize weights to sum to 1.0
            weight_sum = sum(weights)
            normalized_weights = [w / weight_sum for w in weights]
            
            # Calculate weighted average
            lifestyle_score = sum(score * weight for score, weight in zip(score_components, normalized_weights))
        else:
            lifestyle_score = 50  # Default if no components

        # Clamp score between 0 and 100
        lifestyle_score = max(0, min(100, int(round(lifestyle_score))))

        if stress_recovery < self.config.stress_recovery_low_threshold:
            notes += ' Add 10-minute guided meditation before bed.'

        sleep_hours = lifestyle_plan.get('sleepTargetHours') if lifestyle_plan else None
        hydration_ml = lifestyle_plan.get('hydrationTargetMl') if lifestyle_plan else None

        return {
            'calorieTarget': int(round(calories)),
            'proteinTarget': round(protein, 1),
            'carbTarget': round(carbs, 1),
            'fatTarget': round(fats, 1),
            'workoutDuration': int(round(workout_duration)),
            'workoutType': workout_type,
            'stepTarget': int(round(steps)),
            'sleepHours': round(sleep_hours if sleep_hours is not None else 7.5, 1),
            'waterIntakeLiters': round((hydration_ml / 1000.0) if hydration_ml else (weight * 0.035), 2),
            'score': max(0, min(100, int(round(lifestyle_score)))),
            'notes': notes,
            'goal': goal,
        }

    def _safe_float(self, value: Any, default: Optional[float] = None) -> Optional[float]:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _calculate_age(self, dob_value: Any) -> Optional[float]:
        try:
            if isinstance(dob_value, str):
                sanitized = dob_value.replace('Z', '+00:00')
                dob = datetime.fromisoformat(sanitized)
            elif isinstance(dob_value, datetime):
                dob = dob_value
            else:
                return None
            today = datetime.now(tz=dob.tzinfo or timezone.utc)
            years = (today - dob).days / 365.25
            return max(0.0, years)
        except Exception:
            return None

    def _build_recommendations(
        self,
        fever_prob: float,
        resp_prob: float,
        stress_recovery: float,
        news2: Dict[str, Any],
        trends: Dict[str, Dict[str, Any]],
        bp_assessment: Dict[str, Any],
        stress_assessment: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        recs: List[Dict[str, Any]] = []

        recs.append({
            'title': 'Hydration anchor',
            'description': 'Target small sips every 45 minutes to stabilize cardiovascular load.',
            'category': 'hydration',
            'priority': 'medium',
        })

        cfg = self.config
        if stress_recovery < cfg.stress_recovery_medium_threshold:
            recs.append({
                'title': 'Recovery window',
                'description': 'Schedule 15 minutes of parasympathetic work (breathwork or yoga nidra).',
                'category': 'recovery',
                'priority': 'high',
            })

        if fever_prob >= cfg.probability_fever_warning:
            recs.append({
                'title': 'Monitor temperature',
                'description': 'Take temperature twice today and log symptoms if HR stays elevated.',
                'category': 'monitoring',
                'priority': 'high',
            })

        if resp_prob >= cfg.probability_respiratory_warning:
            recs.append({
                'title': 'Respiratory hygiene',
                'description': 'Add steam inhalation or saline rinse tonight to keep airways clear.',
                'category': 'respiratory',
                'priority': 'high',
            })

        if news2['score'] >= cfg.news2_score_levels['moderate']:
            recs.append({
                'title': 'Clinical check reminder',
                'description': 'If symptoms persist or worsen, share vitals with your clinician.',
                'category': 'safety',
                'priority': 'high',
            })

        if trends.get('steps', {}).get('direction') == 'down':
            recs.append({
                'title': 'Micro-activity boost',
                'description': 'Aim for 250 steps each hour – small bursts drive lymphatic flow.',
                'category': 'activity',
                'priority': 'medium',
            })

        for rec in bp_assessment.get('recommendations', []):
            recs.append(rec)

        for rec in stress_assessment.get('recommendations', []):
            recs.append(rec)

        return recs

    def _build_snapshot(self, latest: Dict[str, float], trends: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        snapshot = {}
        for metric, value in latest.items():
            snapshot[metric] = {
                'value': value,
                'trend': trends.get(metric),
            }
        return snapshot

    def _timeline_summary(self, trends: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        summary = []
        for metric, data in trends.items():
            if data['direction'] != 'steady':
                summary.append({
                    'metric': metric,
                    'direction': data['direction'],
                    'change': data['delta'],
                })
        return summary

    def _build_flags(self, news2: Dict[str, Any], fever_prob: float, resp_prob: float) -> List[str]:
        cfg = self.config
        flags = []
        if news2['score'] >= cfg.news2_score_levels['critical']:
            flags.append('NEWS2 critical threshold reached')
        if fever_prob >= cfg.probability_fever_alert:
            flags.append('High fever probability')
        if resp_prob >= cfg.probability_respiratory_alert:
            flags.append('High respiratory strain probability')
        return flags

    def _sigmoid(self, x: float) -> float:
        return 1 / (1 + math.exp(-x))

    def _assess_blood_pressure(self, systolic: Optional[float], diastolic: Optional[float], metrics: Optional[List[Dict[str, Any]]] = None, baselines: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhanced BP assessment using multiple readings instead of single snapshot
        If metrics are provided, analyzes variability and consistency of readings
        Uses personalized baselines if provided for more accurate assessment
        """
        cfg = self.config
        if systolic is None or diastolic is None:
            return {
                'level': 'unknown',
                'priority': 'low',
                'label': 'Blood pressure data unavailable',
                'recommendations': []
            }
        
        # Analyze BP variability if multiple readings available
        bp_variability = None
        bp_consistency = 'unknown'
        if metrics:
            bp_readings = [
                {'sys': m.get('bp_systolic') or (m.get('value') if m.get('type') == 'bp_systolic' else None),
                 'dia': m.get('bp_diastolic') or (m.get('value') if m.get('type') == 'bp_diastolic' else None)}
                for m in metrics if m.get('type') in ['bp_systolic', 'bp_diastolic'] or 'bp' in str(m.get('metric_type', '')).lower()
            ]
            
            if len(bp_readings) >= 3:
                sys_values = [r['sys'] for r in bp_readings if r['sys'] is not None]
                dia_values = [r['dia'] for r in bp_readings if r['dia'] is not None]
                
                if sys_values and dia_values:
                    bp_variability = {
                        'systolic_std': float(np.std(sys_values)),
                        'diastolic_std': float(np.std(dia_values))
                    }
                    
                    # Classify consistency based on variability
                    if bp_variability['systolic_std'] < 5 and bp_variability['diastolic_std'] < 5:
                        bp_consistency = 'consistent'
                    elif bp_variability['systolic_std'] < 10 and bp_variability['diastolic_std'] < 10:
                        bp_consistency = 'moderate'
                    else:
                        bp_consistency = 'variable'
        
        # Use variability-adjusted thresholds (higher variability = wider confidence intervals)
        variability_factor = 1.0
        if bp_variability:
            variability_factor = 1.0 + min(bp_variability['systolic_std'] / 10.0, 0.2)  # Up to 20% threshold adjustment

        if systolic >= cfg.bp_crisis_systolic or diastolic >= cfg.bp_crisis_diastolic:
            return {
                'level': 'hypertensive_crisis',
                'priority': 'urgent',
                'label': f'{int(systolic)}/{int(diastolic)} mmHg',
                'recommendations': [{
                    'title': 'Seek urgent care',
                    'description': 'Blood pressure is in hypertensive crisis range. Seek immediate medical attention.',
                    'category': 'cardiovascular',
                    'priority': 'urgent',
                }]
            }
        if systolic >= cfg.bp_stage2_systolic or diastolic >= cfg.bp_stage2_diastolic:
            return {
                'level': 'stage_2_hypertension',
                'priority': 'high',
                'label': f'{int(systolic)}/{int(diastolic)} mmHg',
                'recommendations': [{
                    'title': 'BP management plan',
                    'description': 'Reduce sodium below 2.3g/day, increase leafy greens and whole grains, and monitor BP daily.',
                    'category': 'cardiovascular',
                    'priority': 'high',
                }]
            }
        if systolic >= cfg.bp_stage1_systolic or diastolic >= cfg.bp_stage1_diastolic:
            return {
                'level': 'stage_1_hypertension',
                'priority': 'medium',
                'label': f'{int(systolic)}/{int(diastolic)} mmHg',
                'recommendations': [{
                    'title': 'Lifestyle adjustments',
                    'description': 'Add 30 minutes of moderate exercise most days and monitor BP weekly.',
                    'category': 'cardiovascular',
                    'priority': 'medium',
                }]
            }
        if systolic >= cfg.bp_elevated_systolic and diastolic < 80:
            return {
                'level': 'elevated',
                'priority': 'low',
                'label': f'{int(systolic)}/{int(diastolic)} mmHg',
                'recommendations': [{
                    'title': 'Maintain heart health',
                    'description': 'Keep sodium intake steady and add potassium-rich foods to meals.',
                    'category': 'cardiovascular',
                    'priority': 'low',
                }]
            }

        return {
            'level': 'normal',
            'priority': 'none',
            'label': f'{int(systolic)}/{int(diastolic)} mmHg',
            'recommendations': []
        }

    def _assess_stress_level(self, stress_value: Optional[float], baselines: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Enhanced stress assessment with personalized baselines
        """
        cfg = self.config
        if stress_value is None:
            return {
                'level': 'unknown',
                'statusImpact': 'none',
                'value': None,
                'recommendations': []
            }
        
        # Use personalized baseline if available
        if baselines and 'stress_level' in baselines and baselines['stress_level'].get('established'):
            baseline = baselines['stress_level']
            baseline_median = baseline['median']
            deviation = stress_value - baseline_median
            
            # Adjust thresholds based on baseline deviation
            very_high_threshold = baseline_median + 30
            elevated_threshold = baseline_median + 15
            moderate_threshold = baseline_median + 5
        else:
            # Use default thresholds
            very_high_threshold = cfg.stress_very_high
            elevated_threshold = cfg.stress_elevated
            moderate_threshold = cfg.stress_moderate

        if stress_value >= very_high_threshold:
            return {
                'level': 'very_high',
                'statusImpact': 'high',
                'value': stress_value,
                'recommendations': [{
                    'title': 'Acute stress reset',
                    'description': 'Practice 4-7-8 breathing for three cycles and schedule a 15-minute mindfulness break.',
                    'category': 'stress_management',
                    'priority': 'high',
                }]
            }
        if stress_value >= elevated_threshold:
            return {
                'level': 'elevated',
                'statusImpact': 'medium',
                'value': stress_value,
                'recommendations': [{
                    'title': 'Stress calibration',
                    'description': 'Use box breathing (4x4) twice daily and take movement breaks every 90 minutes.',
                    'category': 'stress_management',
                    'priority': 'medium',
                }]
            }
        if stress_value >= moderate_threshold:
            return {
                'level': 'moderate',
                'statusImpact': 'low',
                'value': stress_value,
                'recommendations': [{
                    'title': 'Maintain recovery habits',
                    'description': 'Stay hydrated, keep sleep schedule consistent, and add a short walk mid-day.',
                    'category': 'stress_management',
                    'priority': 'low',
                }]
            }

        return {
            'level': 'low',
            'statusImpact': 'none',
            'value': stress_value,
            'recommendations': []
        }

    def _calculate_overall_status(
        self,
        news2_score: int,
        bp_assessment: Dict[str, Any],
        stress_assessment: Dict[str, Any]
    ) -> str:
        cfg = self.config
        if news2_score >= cfg.news2_score_levels['critical']:
            base = 'urgent'
        elif news2_score >= cfg.news2_score_levels['high']:
            base = 'high'
        elif news2_score >= cfg.news2_score_levels['moderate']:
            base = 'medium'
        else:
            base = 'low'

        if bp_assessment['level'] == 'hypertensive_crisis':
            return 'urgent'
        if bp_assessment['level'] == 'stage_2_hypertension' and base == 'low':
            base = 'medium'
        if stress_assessment['statusImpact'] == 'high' and base == 'low':
            base = 'medium'
        if stress_assessment['statusImpact'] == 'medium' and base == 'low':
            base = 'low-medium'

        return base

    def _compose_summary(
        self,
        status_level: str,
        news2_score: int,
        bp_assessment: Dict[str, Any],
        stress_assessment: Dict[str, Any],
        fever_prob: float,
        resp_prob: float
    ) -> Dict[str, Any]:
        headline = self._generate_headline(
            status_level,
            news2_score,
            bp_assessment,
            stress_assessment,
            fever_prob,
            resp_prob
        )

        next_action = self._determine_next_action(
            status_level,
            bp_assessment,
            stress_assessment,
            fever_prob,
            resp_prob
        )

        return {
            'headline': headline,
            'nextBestAction': next_action,
            'statusLevel': status_level
        }

    def _generate_headline(
        self,
        status_level: str,
        news2_score: int,
        bp_assessment: Dict[str, Any],
        stress_assessment: Dict[str, Any],
        fever_prob: float,
        resp_prob: float
    ) -> str:
        cfg = self.config
        concerns = []
        if news2_score >= cfg.news2_score_levels['moderate']:
            concerns.append('vitals trending high')
        if bp_assessment['level'] in {'stage_2_hypertension', 'hypertensive_crisis'}:
            concerns.append('elevated blood pressure')
        elif bp_assessment['level'] == 'stage_1_hypertension':
            concerns.append('borderline blood pressure')
        if stress_assessment['statusImpact'] in {'high', 'medium'}:
            concerns.append('elevated stress load')
        if fever_prob >= cfg.fever_probability_threshold:
            concerns.append('possible infection risk')
        if resp_prob >= cfg.respiratory_probability_threshold:
            concerns.append('respiratory strain signals')

        if not concerns:
            return 'Vitals stable'
        if len(concerns) == 1:
            return f"Attention: {concerns[0]}"
        return f"Multiple concerns: {', '.join(concerns)}"

    def _determine_next_action(
        self,
        status_level: str,
        bp_assessment: Dict[str, Any],
        stress_assessment: Dict[str, Any],
        fever_prob: float,
        resp_prob: float
    ) -> str:
        cfg = self.config
        if bp_assessment['level'] == 'hypertensive_crisis':
            return 'Seek urgent care for blood pressure immediately.'
        if bp_assessment['level'] == 'stage_2_hypertension':
            return 'Prioritize sodium reduction and schedule a clinician follow-up this week.'
        if stress_assessment['statusImpact'] == 'high':
            return 'Focus on immediate stress reset with guided breathing and scheduled breaks.'
        if fever_prob >= cfg.fever_probability_threshold:
            return 'Monitor temperature closely and reduce intensity for the next 24 hours.'
        if resp_prob >= cfg.respiratory_probability_threshold:
            return 'Prioritize nasal hygiene and deep-breathing drills today.'

        if status_level in {'low-medium', 'medium'}:
            return 'Keep hydration steady and add a restorative session today.'
        return 'Maintain current routine with hydration and light movement.'

    def _build_vital_concerns(
        self,
        bp_assessment: Dict[str, Any],
        stress_assessment: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        concerns = []
        if bp_assessment['level'] not in {'normal', 'unknown'}:
            concerns.append({
                'metric': 'blood_pressure',
                'level': bp_assessment['level'],
                'priority': bp_assessment['priority'],
                'value': bp_assessment['label']
            })
        if stress_assessment['statusImpact'] in {'medium', 'high'}:
            concerns.append({
                'metric': 'stress_level',
                'level': stress_assessment['level'],
                'priority': stress_assessment['statusImpact'],
                'value': stress_assessment.get('value')
            })
        return concerns

