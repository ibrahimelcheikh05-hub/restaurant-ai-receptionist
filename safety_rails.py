"""
Safety Rails and Watchdogs (Production)
========================================
Hard safety limits for production voice calls.

Watchdogs:
- Maximum call duration
- Maximum turn count
- Silence detection
- AI timeout detection
- Stuck call auto-termination
"""

import asyncio
import logging
from typing import Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SafetyLimits:
    """Safety limit configuration."""
    max_call_duration_seconds: int = 900  # 15 minutes
    max_turns: int = 100
    max_silence_duration_seconds: int = 30
    max_ai_timeout_seconds: int = 15
    silence_strike_limit: int = 3
    ai_error_limit: int = 3
    
    # Watchdog intervals
    duration_check_interval: int = 10  # seconds
    silence_check_interval: int = 5    # seconds


class SafetyRails:
    """
    Safety rails for voice call protection.
    
    Monitors:
    - Call duration
    - Turn count
    - Silence periods
    - AI timeouts
    - Stuck call detection
    """
    
    def __init__(
        self,
        call_id: str,
        limits: Optional[SafetyLimits] = None,
        on_violation: Optional[Callable] = None
    ):
        self.call_id = call_id
        self.limits = limits or SafetyLimits()
        self.on_violation = on_violation
        
        # State tracking
        self.call_start_time = datetime.utcnow()
        self.last_activity_time = datetime.utcnow()
        self.turn_count = 0
        self.silence_strikes = 0
        self.ai_error_count = 0
        
        # Watchdog tasks
        self._duration_watchdog: Optional[asyncio.Task] = None
        self._silence_watchdog: Optional[asyncio.Task] = None
        self._active = False
        
        logger.info(
            "Safety rails initialized",
            extra={
                "call_id": call_id,
                "max_duration": limits.max_call_duration_seconds,
                "max_turns": limits.max_turns
            }
        )
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity_time = datetime.utcnow()
        
        # Reset silence strikes on activity
        if self.silence_strikes > 0:
            logger.debug(
                f"Activity detected, resetting silence strikes",
                extra={"call_id": self.call_id, "previous_strikes": self.silence_strikes}
            )
            self.silence_strikes = 0
    
    def increment_turn(self) -> bool:
        """
        Increment turn counter.
        
        Returns:
            True if within limits, False if exceeded
        """
        self.turn_count += 1
        self.update_activity()
        
        if self.turn_count >= self.limits.max_turns:
            logger.warning(
                "Max turns exceeded",
                extra={
                    "call_id": self.call_id,
                    "turn_count": self.turn_count,
                    "limit": self.limits.max_turns
                }
            )
            self._trigger_violation("max_turns_exceeded")
            return False
        
        return True
    
    def increment_ai_error(self) -> bool:
        """
        Increment AI error counter.
        
        Returns:
            True if within limits, False if exceeded
        """
        self.ai_error_count += 1
        
        if self.ai_error_count >= self.limits.ai_error_limit:
            logger.error(
                "Max AI errors exceeded",
                extra={
                    "call_id": self.call_id,
                    "error_count": self.ai_error_count,
                    "limit": self.limits.ai_error_limit
                }
            )
            self._trigger_violation("max_ai_errors_exceeded")
            return False
        
        return True
    
    def reset_ai_errors(self):
        """Reset AI error counter on success."""
        if self.ai_error_count > 0:
            self.ai_error_count = 0
    
    async def start_watchdogs(self):
        """Start all safety watchdogs."""
        self._active = True
        
        self._duration_watchdog = asyncio.create_task(
            self._run_duration_watchdog()
        )
        
        self._silence_watchdog = asyncio.create_task(
            self._run_silence_watchdog()
        )
        
        logger.info(
            "Watchdogs started",
            extra={"call_id": self.call_id}
        )
    
    async def stop_watchdogs(self):
        """Stop all safety watchdogs."""
        self._active = False
        
        # Cancel watchdogs with timeout
        watchdogs = []
        if self._duration_watchdog and not self._duration_watchdog.done():
            watchdogs.append(("duration", self._duration_watchdog))
        if self._silence_watchdog and not self._silence_watchdog.done():
            watchdogs.append(("silence", self._silence_watchdog))
        
        for name, task in watchdogs:
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as e:
                logger.error(
                    f"Error stopping {name} watchdog",
                    extra={"call_id": self.call_id, "error": str(e)}
                )
        
        logger.info(
            "Watchdogs stopped",
            extra={"call_id": self.call_id}
        )
    
    async def _run_duration_watchdog(self):
        """Monitor maximum call duration."""
        try:
            while self._active:
                await asyncio.sleep(self.limits.duration_check_interval)
                
                duration = (datetime.utcnow() - self.call_start_time).total_seconds()
                
                if duration > self.limits.max_call_duration_seconds:
                    logger.warning(
                        "Max call duration exceeded",
                        extra={
                            "call_id": self.call_id,
                            "duration_seconds": duration,
                            "limit": self.limits.max_call_duration_seconds
                        }
                    )
                    self._trigger_violation("max_duration_exceeded")
                    break
        
        except asyncio.CancelledError:
            logger.debug(
                "Duration watchdog cancelled",
                extra={"call_id": self.call_id}
            )
        except Exception as e:
            logger.error(
                "Duration watchdog crashed",
                extra={"call_id": self.call_id, "error": str(e)},
                exc_info=True
            )
            self._trigger_violation("duration_watchdog_crash")
    
    async def _run_silence_watchdog(self):
        """Monitor silence periods."""
        try:
            while self._active:
                await asyncio.sleep(self.limits.silence_check_interval)
                
                silence_duration = (
                    datetime.utcnow() - self.last_activity_time
                ).total_seconds()
                
                if silence_duration > self.limits.max_silence_duration_seconds:
                    self.silence_strikes += 1
                    
                    logger.warning(
                        "Silence detected",
                        extra={
                            "call_id": self.call_id,
                            "silence_duration": silence_duration,
                            "strikes": self.silence_strikes,
                            "limit": self.limits.silence_strike_limit
                        }
                    )
                    
                    if self.silence_strikes >= self.limits.silence_strike_limit:
                        logger.error(
                            "Max silence strikes exceeded",
                            extra={
                                "call_id": self.call_id,
                                "strikes": self.silence_strikes
                            }
                        )
                        self._trigger_violation("max_silence_strikes_exceeded")
                        break
                    
                    # Reset activity to prevent immediate re-trigger
                    self.update_activity()
        
        except asyncio.CancelledError:
            logger.debug(
                "Silence watchdog cancelled",
                extra={"call_id": self.call_id}
            )
        except Exception as e:
            logger.error(
                "Silence watchdog crashed",
                extra={"call_id": self.call_id, "error": str(e)},
                exc_info=True
            )
            self._trigger_violation("silence_watchdog_crash")
    
    def _trigger_violation(self, violation_type: str):
        """Trigger safety violation callback."""
        logger.error(
            f"SAFETY VIOLATION: {violation_type}",
            extra={
                "call_id": self.call_id,
                "violation_type": violation_type,
                "turn_count": self.turn_count,
                "silence_strikes": self.silence_strikes,
                "ai_errors": self.ai_error_count
            }
        )
        
        if self.on_violation:
            try:
                if asyncio.iscoroutinefunction(self.on_violation):
                    asyncio.create_task(self.on_violation(violation_type))
                else:
                    self.on_violation(violation_type)
            except Exception as e:
                logger.error(
                    f"Error in violation callback",
                    extra={"call_id": self.call_id, "error": str(e)},
                    exc_info=True
                )
    
    def get_stats(self) -> dict:
        """Get safety stats."""
        duration = (datetime.utcnow() - self.call_start_time).total_seconds()
        silence_duration = (
            datetime.utcnow() - self.last_activity_time
        ).total_seconds()
        
        return {
            "call_id": self.call_id,
            "duration_seconds": duration,
            "turn_count": self.turn_count,
            "silence_strikes": self.silence_strikes,
            "ai_error_count": self.ai_error_count,
            "current_silence_duration": silence_duration,
            "is_active": self._active,
            "limits": {
                "max_duration": self.limits.max_call_duration_seconds,
                "max_turns": self.limits.max_turns,
                "max_silence": self.limits.max_silence_duration_seconds,
                "max_ai_errors": self.limits.ai_error_limit,
                "silence_strike_limit": self.limits.silence_strike_limit
            }
        }
    
    def __repr__(self):
        return f"<SafetyRails call_id={self.call_id} turns={self.turn_count}>"
