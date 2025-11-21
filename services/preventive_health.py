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
        if not metrics:
            raise ValueError("metrics payload is required")

        if lookback_days is None or lookback_days <= 0:
            lookback_days = self.config.default_lookback_days

        time_series = self._prepare_time_series(metrics, lookback_days)
        latest = {metric: points[-1]['value'] for metric, points in time_series.items() if points}
        if not latest:
            raise ValueError("no valid metrics available for analysis")

        trends = {
            metric: self._calculate_trend(points)
            for metric, points in time_series.items()
            if points
        }

        news2 = self._calculate_news2(latest)
        fever_prob, fever_signals = self._estimate_fever_probability(latest, trends, news2['score'])
        resp_prob, resp_signals = self._estimate_respiratory_probability(latest, trends, news2['score'])
        stress_recovery = self._estimate_stress_recovery(latest, trends)

        bp_assessment = self._assess_blood_pressure(
            latest.get('bp_systolic'),
            latest.get('bp_diastolic')
        )
        stress_assessment = self._assess_stress_level(latest.get('stress_level'))
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
            },
        }
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _prepare_time_series(
        self,
        metrics: List[Dict[str, Any]],
        lookback_days: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        lookback_start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for entry in metrics:
            metric_type = entry.get('type') or entry.get('metric_type')
            if not metric_type:
                continue
            metric_key = self.METRIC_ALIASES.get(metric_type.lower())
            if not metric_key:
                continue

            value = entry.get('value')
            if value is None:
                continue

            try:
                value = float(value)
            except (TypeError, ValueError):
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
        if len(points) < 2:
            return {'direction': 'steady', 'delta': 0.0, 'changePerDay': 0.0}

        values = [p['value'] for p in points]
        timestamps = [p['timestamp'] for p in points]
        total_days = max((timestamps[-1] - timestamps[0]).total_seconds() / 86400.0, 1e-6)
        delta = values[-1] - values[0]
        change_per_day = delta / total_days
        direction = 'up' if delta > 0.5 else 'down' if delta < -0.5 else 'steady'

        return {
            'direction': direction,
            'delta': round(delta, 2),
            'changePerDay': round(change_per_day, 2),
        }

    def _calculate_news2(self, latest: Dict[str, float]) -> Dict[str, Any]:
        score = 0
        breakdown: List[Dict[str, Any]] = []
        cfg = self.config

        resp_rate = latest.get('respiratory_rate')
        resp_points = 0
        if resp_rate is not None:
            if resp_rate <= cfg.news2_respiratory['very_low']:
                resp_points = 3
            elif resp_rate <= cfg.news2_respiratory['low']:
                resp_points = 1
            elif resp_rate <= cfg.news2_respiratory['normal_high']:
                resp_points = 0
            elif resp_rate <= cfg.news2_respiratory['high']:
                resp_points = 2
            else:
                resp_points = 3
        breakdown.append({'metric': 'respiratory_rate', 'value': resp_rate, 'points': resp_points})
        score += resp_points

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
            if heart_rate <= cfg.news2_heart_rate['very_low']:
                heart_points = 3
            elif heart_rate <= cfg.news2_heart_rate['low']:
                heart_points = 1
            elif heart_rate <= cfg.news2_heart_rate['normal_high']:
                heart_points = 0
            elif heart_rate <= cfg.news2_heart_rate['high']:
                heart_points = 1
            elif heart_rate <= cfg.news2_heart_rate['very_high']:
                heart_points = 2
            else:
                heart_points = 3
        breakdown.append({'metric': 'heart_rate', 'value': heart_rate, 'points': heart_points})
        score += heart_points

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

    def _estimate_fever_probability(
        self,
        latest: Dict[str, float],
        trends: Dict[str, Dict[str, Any]],
        news2_score: int
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

        trend_bonus = 0.0
        if trends.get('heart_rate', {}).get('direction') == 'up':
            trend_bonus += cfg.fever_trend_bonus['heart_rate']
        if trends.get('temperature', {}).get('direction') == 'up':
            trend_bonus += cfg.fever_trend_bonus['temperature']

        score = (
            cfg.fever_weights['temperature'] * temp_signal_value
            + cfg.fever_weights['heart_rate'] * hr_signal
            + cfg.fever_weights['stress'] * stress_signal
            + cfg.fever_weights['spo2'] * spo2_signal
            + trend_bonus
            + news2_score / 12.0
        )
        probability = self._sigmoid(score - cfg.fever_sigmoid_offset)

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
        news2_score: int
    ) -> Tuple[float, List[str]]:
        cfg = self.config
        resp_rate = latest.get('respiratory_rate', 16)
        spo2 = latest.get('oxygen_saturation', 98)
        stress = latest.get('stress_level', 40)
        hr = latest.get('heart_rate', 72)

        resp_signal = max(0.0, (resp_rate - cfg.respiratory_rate_high) / 6.0)
        spo2_signal = max(0.0, (cfg.respiratory_spo2_threshold - spo2) / 4.0)
        hr_signal = max(0.0, (hr - cfg.respiratory_hr_threshold) / 25.0)
        stress_signal = max(0.0, (stress - cfg.respiratory_stress_threshold) / 25.0)

        trend_bonus = 0.0
        if trends.get('respiratory_rate', {}).get('direction') == 'up':
            trend_bonus += cfg.respiratory_trend_bonus['respiratory_rate']
        if trends.get('oxygen_saturation', {}).get('direction') == 'down':
            trend_bonus += cfg.respiratory_trend_bonus['oxygen_saturation']

        score = (
            cfg.respiratory_weights['respiratory_rate'] * resp_signal
            + cfg.respiratory_weights['spo2'] * spo2_signal
            + cfg.respiratory_weights['heart_rate'] * hr_signal
            + cfg.respiratory_weights['stress'] * stress_signal
            + trend_bonus
            + news2_score / 15.0
        )
        probability = self._sigmoid(score - cfg.respiratory_sigmoid_offset)

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
        trends: Dict[str, Dict[str, Any]]
    ) -> float:
        cfg = self.config
        stress = latest.get('stress_level', 40)
        heart_rate = latest.get('heart_rate', 72)
        respiratory = latest.get('respiratory_rate', 16)

        stress_norm = (100 - stress) / 100.0
        hr_norm = max(0.0, 1 - abs(heart_rate - cfg.stress_recovery_optimal_hr) / 40.0)
        rr_norm = max(0.0, 1 - abs(respiratory - cfg.stress_recovery_optimal_rr) / 10.0)

        trend_bonus = cfg.stress_recovery_trend_bonus if trends.get('stress_level', {}).get('direction') == 'down' else 0.0

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
        cfg = self.config
        hydration_target = cfg.lifestyle_hydration_base_ml
        if user_profile:
            weight = user_profile.get('weightKg')
            if weight:
                hydration_target = int(weight * cfg.lifestyle_hydration_ml_per_kg)

        sleep_target = cfg.lifestyle_sleep_target_hours
        if user_profile and user_profile.get('age') and user_profile['age'] < cfg.lifestyle_young_adult_age:
            sleep_target = cfg.lifestyle_sleep_young_adult_hours

        plan = {
            'hydrationTargetMl': hydration_target,
            'sleepTargetHours': sleep_target,
            'morning': [
                "2 glasses of water on wake-up",
                "5-minute box-breathing while seated",
            ],
            'afternoon': [
                "10-minute brisk walk to keep HRV responsive",
                "Lunch with leafy greens + lean protein",
            ],
            'evening': [
                "Screen off 45 min before bed",
                "Light stretching / mobility to downshift stress",
            ],
        }

        if stress_recovery < cfg.stress_recovery_low_threshold:
            plan['evening'].append("Add 10-minute guided meditation before bed")

        if trends.get('heart_rate', {}).get('direction') == 'up':
            plan['morning'].append("Swap caffeine for herbal tea today")

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

        calories = tdee
        protein = weight * 1.6
        carbs = calories * 0.4 / 4
        fats = calories * 0.3 / 9
        workout_duration = 30
        workout_type = 'General Exercise'
        steps = 8000
        notes = 'General health and wellness recommendations.'

        if goal == 'weight_loss':
            calories = tdee * 0.85
            protein = weight * 2.2
            carbs = calories * 0.35 / 4
            fats = calories * 0.25 / 9
            workout_duration = 45
            workout_type = 'Cardio + Strength Training'
            steps = 10000
            notes = 'Focus on calorie deficit with high protein intake. Include both cardio and strength training.'
        elif goal == 'weight_gain':
            calories = tdee * 1.15
            protein = weight * 1.8
            carbs = calories * 0.45 / 4
            fats = calories * 0.25 / 9
            workout_duration = 60
            workout_type = 'Strength Training + Cardio'
            steps = 8000
            notes = 'Calorie surplus with balanced macros. Focus on progressive strength training.'
        elif goal == 'muscle_gain':
            calories = tdee * 1.1
            protein = weight * 2.5
            carbs = calories * 0.40 / 4
            fats = calories * 0.20 / 9
            workout_duration = 60
            workout_type = 'Strength Training'
            steps = 7000
            notes = 'High protein intake essential for muscle growth. Focus on compound movements.'
        elif goal == 'maintain':
            calories = tdee
            protein = weight * 1.6
            carbs = calories * 0.40 / 4
            fats = calories * 0.30 / 9
            workout_duration = 30
            workout_type = 'Mixed Training'
            steps = 8000
            notes = 'Maintain current weight with balanced nutrition and regular exercise.'
        elif goal == 'improve_endurance':
            calories = tdee * 1.05
            protein = weight * 1.6
            carbs = calories * 0.50 / 4
            fats = calories * 0.25 / 9
            workout_duration = 60
            workout_type = 'Cardio + Endurance Training'
            steps = 12000
            notes = 'Higher carb intake for sustained energy. Focus on cardiovascular and endurance exercises.'

        # Calculate dynamic lifestyle score based on multiple factors
        lifestyle_score = 50  # Base score
        
        # BMI contribution (0-20 points)
        if 18.5 <= bmi <= 24.9:
            lifestyle_score += 20
        elif 24.9 < bmi <= 27:
            lifestyle_score += 10
        elif 27 < bmi <= 30:
            lifestyle_score += 5
        elif bmi < 18.5:
            lifestyle_score += 5
        else:  # bmi > 30
            lifestyle_score -= 5
        
        # Stress recovery contribution (0-20 points)
        stress_recovery_points = stress_recovery * 20
        lifestyle_score += stress_recovery_points
        
        # Vital signs contribution (0-15 points)
        heart_rate = latest.get('heart_rate', 72)
        respiratory_rate = latest.get('respiratory_rate', 16)
        temperature = latest.get('temperature', 36.5)
        spo2 = latest.get('oxygen_saturation', 98)
        
        # Heart rate score (0-5 points)
        if 60 <= heart_rate <= 100:
            lifestyle_score += 5
        elif 50 <= heart_rate < 60 or 100 < heart_rate <= 110:
            lifestyle_score += 3
        else:
            lifestyle_score += 1
        
        # Respiratory rate score (0-3 points)
        if 12 <= respiratory_rate <= 20:
            lifestyle_score += 3
        elif 10 <= respiratory_rate < 12 or 20 < respiratory_rate <= 24:
            lifestyle_score += 2
        else:
            lifestyle_score += 1
        
        # Temperature score (0-3 points)
        if 36.1 <= temperature <= 37.2:
            lifestyle_score += 3
        elif 35.5 <= temperature < 36.1 or 37.2 < temperature <= 37.5:
            lifestyle_score += 2
        else:
            lifestyle_score += 1
        
        # SpO2 score (0-4 points)
        if spo2 >= 98:
            lifestyle_score += 4
        elif spo2 >= 96:
            lifestyle_score += 3
        elif spo2 >= 94:
            lifestyle_score += 2
        else:
            lifestyle_score += 1
        
        # Blood pressure contribution (0-10 points)
        if bp_assessment:
            bp_level = bp_assessment.get('level', 'normal')
            if bp_level == 'normal':
                lifestyle_score += 10
            elif bp_level == 'elevated':
                lifestyle_score += 7
            elif bp_level == 'stage1':
                lifestyle_score += 4
            elif bp_level == 'stage2':
                lifestyle_score += 1
            else:  # crisis
                lifestyle_score -= 5
        
        # Stress level contribution (0-10 points)
        if stress_assessment:
            stress_level = stress_assessment.get('level', 'normal')
            if stress_level == 'low':
                lifestyle_score += 10
            elif stress_level == 'moderate':
                lifestyle_score += 7
            elif stress_level == 'elevated':
                lifestyle_score += 4
            else:  # very_high
                lifestyle_score += 1
        
        # Trends contribution (0-10 points) - improving trends boost score
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
                lifestyle_score += 10
            elif improving_count == declining_count and improving_count > 0:
                lifestyle_score += 5
            elif declining_count > improving_count:
                lifestyle_score -= 5
        
        # NEWS2 score contribution (0-10 points) - lower is better
        if news2:
            news2_score = news2.get('score', 0)
            if news2_score == 0:
                lifestyle_score += 10
            elif news2_score <= 2:
                lifestyle_score += 7
            elif news2_score <= 4:
                lifestyle_score += 4
            elif news2_score <= 6:
                lifestyle_score += 1
            else:  # score >= 7
                lifestyle_score -= 5

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

    def _assess_blood_pressure(self, systolic: Optional[float], diastolic: Optional[float]) -> Dict[str, Any]:
        cfg = self.config
        if systolic is None or diastolic is None:
            return {
                'level': 'unknown',
                'priority': 'low',
                'label': 'Blood pressure data unavailable',
                'recommendations': []
            }

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

    def _assess_stress_level(self, stress_value: Optional[float]) -> Dict[str, Any]:
        cfg = self.config
        if stress_value is None:
            return {
                'level': 'unknown',
                'statusImpact': 'none',
                'value': None,
                'recommendations': []
            }

        if stress_value >= cfg.stress_very_high:
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
        if stress_value >= cfg.stress_elevated:
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
        if stress_value >= cfg.stress_moderate:
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

