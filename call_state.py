"""
Call State Machine (Production)
================================
Formal state transitions for voice call lifecycle.

State invariants:
- AI outputs NEVER directly change system state
- All transitions are deterministic and validated
- State changes are logged and tracked
"""

import logging
from enum import Enum
from typing import Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)


class CallState(Enum):
    """
    Formal call lifecycle states.
    
    State flow:
        INIT -> GREETING -> LISTENING <-> THINKING <-> SPEAKING
                                      -> CLOSING -> CLOSED
    """
    INIT = "init"           # Call created, not yet started
    GREETING = "greeting"   # Playing initial greeting
    LISTENING = "listening" # Waiting for user input
    THINKING = "thinking"   # Processing AI response
    SPEAKING = "speaking"   # Playing AI response
    CLOSING = "closing"     # Graceful shutdown in progress
    CLOSED = "closed"       # Call terminated


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
    pass


class CallStateMachine:
    """
    Manages call state transitions with validation.
    
    Enforces:
    - Valid transition paths only
    - State change logging
    - Transition metrics
    """
    
    # Define valid state transitions
    VALID_TRANSITIONS = {
        CallState.INIT: {CallState.GREETING, CallState.CLOSED},
        CallState.GREETING: {CallState.LISTENING, CallState.CLOSING, CallState.CLOSED},
        CallState.LISTENING: {CallState.THINKING, CallState.CLOSING, CallState.CLOSED},
        CallState.THINKING: {CallState.SPEAKING, CallState.LISTENING, CallState.CLOSING, CallState.CLOSED},
        CallState.SPEAKING: {CallState.LISTENING, CallState.CLOSING, CallState.CLOSED},
        CallState.CLOSING: {CallState.CLOSED},
        CallState.CLOSED: set()  # Terminal state
    }
    
    def __init__(self, call_id: str, initial_state: CallState = CallState.INIT):
        self.call_id = call_id
        self._current_state = initial_state
        self._state_history = [(initial_state, datetime.utcnow())]
        self._transition_count = 0
        
        logger.info(
            "State machine initialized",
            extra={
                "call_id": call_id,
                "initial_state": initial_state.value
            }
        )
    
    @property
    def current_state(self) -> CallState:
        """Get current state."""
        return self._current_state
    
    def can_transition_to(self, target_state: CallState) -> bool:
        """
        Check if transition to target state is valid.
        
        Args:
            target_state: Desired next state
            
        Returns:
            True if transition is valid
        """
        return target_state in self.VALID_TRANSITIONS.get(self._current_state, set())
    
    def transition(self, target_state: CallState, reason: Optional[str] = None) -> bool:
        """
        Attempt state transition with validation.
        
        Args:
            target_state: Desired next state
            reason: Optional reason for transition
            
        Returns:
            True if transition succeeded
            
        Raises:
            StateTransitionError: If transition is invalid
        """
        if not self.can_transition_to(target_state):
            error_msg = (
                f"Invalid transition: {self._current_state.value} -> {target_state.value}"
            )
            logger.error(
                error_msg,
                extra={
                    "call_id": self.call_id,
                    "from_state": self._current_state.value,
                    "to_state": target_state.value,
                    "reason": reason
                }
            )
            raise StateTransitionError(error_msg)
        
        # Perform transition
        old_state = self._current_state
        self._current_state = target_state
        self._transition_count += 1
        self._state_history.append((target_state, datetime.utcnow()))
        
        logger.info(
            f"State transition: {old_state.value} -> {target_state.value}",
            extra={
                "call_id": self.call_id,
                "from_state": old_state.value,
                "to_state": target_state.value,
                "reason": reason,
                "transition_count": self._transition_count
            }
        )
        
        return True
    
    def force_transition(self, target_state: CallState, reason: str):
        """
        Force transition without validation (emergency use only).
        
        Args:
            target_state: Target state
            reason: Reason for forced transition
        """
        logger.warning(
            f"FORCED state transition: {self._current_state.value} -> {target_state.value}",
            extra={
                "call_id": self.call_id,
                "from_state": self._current_state.value,
                "to_state": target_state.value,
                "reason": reason
            }
        )
        
        self._current_state = target_state
        self._transition_count += 1
        self._state_history.append((target_state, datetime.utcnow()))
    
    def is_terminal(self) -> bool:
        """Check if current state is terminal."""
        return self._current_state == CallState.CLOSED
    
    def is_active(self) -> bool:
        """Check if call is in an active conversational state."""
        return self._current_state in {
            CallState.LISTENING,
            CallState.THINKING,
            CallState.SPEAKING
        }
    
    def get_state_duration(self) -> float:
        """Get duration in current state (seconds)."""
        if not self._state_history:
            return 0.0
        
        _, entered_at = self._state_history[-1]
        return (datetime.utcnow() - entered_at).total_seconds()
    
    def get_history(self) -> list:
        """Get state transition history."""
        return [
            {
                "state": state.value,
                "timestamp": ts.isoformat(),
                "duration_seconds": (
                    (self._state_history[i + 1][1] - ts).total_seconds()
                    if i + 1 < len(self._state_history)
                    else (datetime.utcnow() - ts).total_seconds()
                )
            }
            for i, (state, ts) in enumerate(self._state_history)
        ]
    
    def __repr__(self):
        return f"<CallStateMachine call_id={self.call_id} state={self._current_state.value}>"
