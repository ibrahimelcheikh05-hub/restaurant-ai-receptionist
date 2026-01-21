"""
Latency Tracking Module (Enterprise Production)
===============================================
Comprehensive latency monitoring for voice ordering system.

Tracks:
✅ End-to-end call latency
✅ STT latency (speech-to-text)
✅ LLM latency (GPT-4o-mini)
✅ TTS latency (text-to-speech)
✅ Translation latency
✅ Database query latency
✅ API call latency
✅ Turn-around time (user speaks → AI responds)
✅ P50, P95, P99 percentiles
✅ Real-time metrics

Version: 1.0.0 (Enterprise)
Last Updated: 2026-01-21
"""

import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import statistics

try:
    from prometheus_client import Histogram, Counter, Gauge, Summary
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False


logger = logging.getLogger(__name__)


# ============================================================================
# LATENCY CATEGORIES
# ============================================================================

class LatencyType(Enum):
    """Types of latency measurements."""
    STT = "stt"                           # Speech-to-text
    LLM = "llm"                           # Language model
    TTS = "tts"                           # Text-to-speech
    TRANSLATION = "translation"            # Translation
    DATABASE = "database"                  # Database queries
    API_CALL = "api_call"                 # External API calls
    TURN_AROUND = "turn_around"           # User speaks → AI responds
    END_TO_END = "end_to_end"             # Full call duration
    MEMORY_READ = "memory_read"           # Memory operations
    MEMORY_WRITE = "memory_write"         # Memory writes
    ORDER_PROCESSING = "order_processing"  # Order operations
    UPSELL_GENERATION = "upsell"          # Upsell suggestions
    SMS_SEND = "sms"                      # SMS delivery


# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

if METRICS_ENABLED:
    # Histograms for latency distribution
    latency_histogram = Histogram(
        'latency_seconds',
        'Latency measurements',
        ['component', 'operation'],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    
    # Summary for percentiles
    latency_summary = Summary(
        'latency_summary_seconds',
        'Latency summary with percentiles',
        ['component']
    )
    
    # Counter for slow operations
    slow_operations_total = Counter(
        'slow_operations_total',
        'Operations exceeding latency threshold',
        ['component', 'threshold']
    )
    
    # Gauge for current latency
    current_latency_gauge = Gauge(
        'current_latency_seconds',
        'Current latency measurement',
        ['component']
    )
    
    # Turn-around time specific metrics
    turn_around_time = Histogram(
        'turn_around_time_seconds',
        'Time from user input to AI response',
        buckets=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    )


# ============================================================================
# LATENCY MEASUREMENT
# ============================================================================

@dataclass
class LatencyMeasurement:
    """Single latency measurement."""
    component: str
    operation: str
    duration_ms: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        return self.duration_ms / 1000.0
    
    def is_slow(self, threshold_ms: float = 1000) -> bool:
        """Check if measurement exceeds threshold."""
        return self.duration_ms > threshold_ms


class LatencyTracker:
    """
    Tracks and analyzes latency measurements.
    """
    
    def __init__(self, session_id: str, max_history: int = 1000):
        self.session_id = session_id
        self.max_history = max_history
        
        # Store measurements by component
        self.measurements: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        
        # Track active timers
        self.active_timers: Dict[str, float] = {}
        
        # Session stats
        self.session_start = datetime.utcnow()
        self.total_measurements = 0
        
        logger.info(f"LatencyTracker initialized: {session_id}")
    
    def start_timer(self, operation_id: str) -> None:
        """Start timing an operation."""
        self.active_timers[operation_id] = time.time()
    
    def end_timer(
        self,
        operation_id: str,
        component: LatencyType,
        operation: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[LatencyMeasurement]:
        """
        End timing and record measurement.
        
        Args:
            operation_id: Unique ID for this operation
            component: Component type (STT, LLM, etc.)
            operation: Specific operation name
            metadata: Additional context
            
        Returns:
            LatencyMeasurement or None if timer not found
        """
        if operation_id not in self.active_timers:
            logger.warning(f"Timer not found: {operation_id}")
            return None
        
        # Calculate duration
        start_time = self.active_timers.pop(operation_id)
        duration_seconds = time.time() - start_time
        duration_ms = duration_seconds * 1000
        
        # Create measurement
        measurement = LatencyMeasurement(
            component=component.value,
            operation=operation or component.value,
            duration_ms=duration_ms,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        # Store measurement
        self.measurements[component.value].append(measurement)
        self.total_measurements += 1
        
        # Track in Prometheus
        if METRICS_ENABLED:
            latency_histogram.labels(
                component=component.value,
                operation=operation
            ).observe(duration_seconds)
            
            latency_summary.labels(
                component=component.value
            ).observe(duration_seconds)
            
            current_latency_gauge.labels(
                component=component.value
            ).set(duration_seconds)
            
            # Track slow operations
            if measurement.is_slow(1000):  # > 1 second
                slow_operations_total.labels(
                    component=component.value,
                    threshold='1s'
                ).inc()
            
            # Special tracking for turn-around time
            if component == LatencyType.TURN_AROUND:
                turn_around_time.observe(duration_seconds)
        
        # Log slow operations
        if measurement.is_slow(1000):
            logger.warning(
                f"Slow operation detected: {component.value}/{operation} - "
                f"{duration_ms:.1f}ms"
            )
        else:
            logger.debug(
                f"Latency: {component.value}/{operation} - {duration_ms:.1f}ms"
            )
        
        return measurement
    
    def track_latency(
        self,
        component: LatencyType,
        duration_ms: float,
        operation: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> LatencyMeasurement:
        """
        Directly track a latency measurement (without timer).
        
        Args:
            component: Component type
            duration_ms: Duration in milliseconds
            operation: Operation name
            metadata: Additional context
            
        Returns:
            LatencyMeasurement
        """
        measurement = LatencyMeasurement(
            component=component.value,
            operation=operation or component.value,
            duration_ms=duration_ms,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.measurements[component.value].append(measurement)
        self.total_measurements += 1
        
        # Track in Prometheus
        if METRICS_ENABLED:
            duration_seconds = duration_ms / 1000.0
            latency_histogram.labels(
                component=component.value,
                operation=operation
            ).observe(duration_seconds)
        
        return measurement
    
    def get_stats(
        self,
        component: Optional[LatencyType] = None,
        minutes: int = 5
    ) -> Dict[str, Any]:
        """
        Get latency statistics.
        
        Args:
            component: Specific component (or None for all)
            minutes: Time window in minutes
            
        Returns:
            Statistics dictionary
        """
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        if component:
            # Stats for specific component
            measurements = [
                m for m in self.measurements[component.value]
                if m.timestamp > cutoff_time
            ]
            
            if not measurements:
                return {"component": component.value, "count": 0}
            
            durations = [m.duration_ms for m in measurements]
            
            return {
                "component": component.value,
                "count": len(measurements),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "mean_ms": statistics.mean(durations),
                "median_ms": statistics.median(durations),
                "p95_ms": self._percentile(durations, 0.95),
                "p99_ms": self._percentile(durations, 0.99),
                "slow_count": sum(1 for m in measurements if m.is_slow()),
                "slow_percentage": sum(1 for m in measurements if m.is_slow()) / len(measurements) * 100
            }
        else:
            # Stats for all components
            stats = {}
            for comp_name, comp_measurements in self.measurements.items():
                recent = [
                    m for m in comp_measurements
                    if m.timestamp > cutoff_time
                ]
                
                if recent:
                    durations = [m.duration_ms for m in recent]
                    stats[comp_name] = {
                        "count": len(recent),
                        "mean_ms": statistics.mean(durations),
                        "p95_ms": self._percentile(durations, 0.95),
                        "p99_ms": self._percentile(durations, 0.99),
                    }
            
            return stats
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get complete session summary."""
        session_duration = (datetime.utcnow() - self.session_start).total_seconds()
        
        return {
            "session_id": self.session_id,
            "session_duration_seconds": session_duration,
            "total_measurements": self.total_measurements,
            "components_tracked": list(self.measurements.keys()),
            "stats_by_component": self.get_stats(),
            "recent_measurements": self._get_recent_measurements(10)
        }
    
    def _get_recent_measurements(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent measurements across all components."""
        all_measurements = []
        for measurements in self.measurements.values():
            all_measurements.extend(measurements)
        
        # Sort by timestamp
        all_measurements.sort(key=lambda m: m.timestamp, reverse=True)
        
        return [
            {
                "component": m.component,
                "operation": m.operation,
                "duration_ms": m.duration_ms,
                "timestamp": m.timestamp.isoformat(),
                "metadata": m.metadata
            }
            for m in all_measurements[:limit]
        ]


# ============================================================================
# GLOBAL TRACKER REGISTRY
# ============================================================================

_trackers: Dict[str, LatencyTracker] = {}


def get_tracker(session_id: str) -> LatencyTracker:
    """Get or create latency tracker for session."""
    if session_id not in _trackers:
        _trackers[session_id] = LatencyTracker(session_id)
    return _trackers[session_id]


def remove_tracker(session_id: str) -> Optional[Dict[str, Any]]:
    """Remove tracker and return final summary."""
    if session_id in _trackers:
        tracker = _trackers.pop(session_id)
        return tracker.get_summary()
    return None


def get_all_stats() -> Dict[str, Any]:
    """Get stats from all active trackers."""
    return {
        "active_trackers": len(_trackers),
        "trackers": {
            session_id: tracker.get_stats()
            for session_id, tracker in _trackers.items()
        }
    }


# ============================================================================
# CONVENIENCE CONTEXT MANAGER
# ============================================================================

class track_latency:
    """
    Context manager for easy latency tracking.
    
    Usage:
        with track_latency(session_id, LatencyType.STT, "transcribe"):
            # Your code here
            result = await transcribe_audio(audio)
    """
    
    def __init__(
        self,
        session_id: str,
        component: LatencyType,
        operation: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.session_id = session_id
        self.component = component
        self.operation = operation
        self.metadata = metadata
        self.operation_id = f"{component.value}_{int(time.time() * 1000)}"
        self.tracker = get_tracker(session_id)
    
    def __enter__(self):
        self.tracker.start_timer(self.operation_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracker.end_timer(
            self.operation_id,
            self.component,
            self.operation,
            self.metadata
        )


# ============================================================================
# ASYNC CONTEXT MANAGER
# ============================================================================

class track_latency_async:
    """
    Async context manager for latency tracking.
    
    Usage:
        async with track_latency_async(session_id, LatencyType.LLM, "generate"):
            response = await llm.generate(prompt)
    """
    
    def __init__(
        self,
        session_id: str,
        component: LatencyType,
        operation: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.session_id = session_id
        self.component = component
        self.operation = operation
        self.metadata = metadata
        self.operation_id = f"{component.value}_{int(time.time() * 1000)}"
        self.tracker = get_tracker(session_id)
    
    async def __aenter__(self):
        self.tracker.start_timer(self.operation_id)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.tracker.end_timer(
            self.operation_id,
            self.component,
            self.operation,
            self.metadata
        )


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def demo():
        print("Latency Tracking Demo")
        print("=" * 60)
        
        session_id = "demo_session"
        tracker = get_tracker(session_id)
        
        # Example 1: Using context manager
        print("\n1. Using context manager:")
        with track_latency(session_id, LatencyType.STT, "transcribe"):
            await asyncio.sleep(0.3)  # Simulate 300ms STT
        
        # Example 2: Using async context manager
        print("2. Using async context manager:")
        async with track_latency_async(session_id, LatencyType.LLM, "generate"):
            await asyncio.sleep(0.5)  # Simulate 500ms LLM
        
        # Example 3: Manual timing
        print("3. Manual timing:")
        tracker.start_timer("tts_1")
        await asyncio.sleep(0.2)  # Simulate 200ms TTS
        tracker.end_timer("tts_1", LatencyType.TTS, "synthesize")
        
        # Example 4: Direct tracking
        print("4. Direct tracking:")
        tracker.track_latency(LatencyType.DATABASE, 50, "query_orders")
        
        # Get stats
        print("\n" + "=" * 60)
        print("STATISTICS")
        print("=" * 60)
        
        stats = tracker.get_stats()
        for component, component_stats in stats.items():
            print(f"\n{component.upper()}:")
            print(f"  Count: {component_stats['count']}")
            print(f"  Mean: {component_stats['mean_ms']:.1f}ms")
            print(f"  P95: {component_stats['p95_ms']:.1f}ms")
            print(f"  P99: {component_stats['p99_ms']:.1f}ms")
        
        # Summary
        print("\n" + "=" * 60)
        summary = tracker.get_summary()
        print(f"Session: {summary['session_id']}")
        print(f"Duration: {summary['session_duration_seconds']:.1f}s")
        print(f"Total Measurements: {summary['total_measurements']}")
        print("=" * 60)
    
    asyncio.run(demo())
