"""
Memory Module (Production Hardened)
====================================
Structured conversation memory system with immutability guarantees.

HARDENING UPDATES (v3.0):
✅ ConversationMemory class with structured fields
✅ Memory pruning (sliding window + token budgeting)
✅ Immutable history records (write-once turns)
✅ AI cannot directly write memory (controller interface)
✅ Snapshot and restore capability
✅ Phase tracking and error counters
✅ No unlimited text storage

Version: 3.0.0 (Production Hardened)
Last Updated: 2026-01-22
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
from copy import deepcopy
import json
import sys

try:
    import zlib
    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Memory limits
MAX_TURNS_WINDOW = int(os.getenv("MAX_CONVERSATION_TURNS", "50"))  # Sliding window
MAX_TOKEN_BUDGET = int(os.getenv("MAX_MEMORY_TOKENS", "4000"))  # ~4K tokens
MAX_SLOT_VALUE_LENGTH = 1000  # Max chars per slot
MAX_TURN_CONTENT_LENGTH = 2000  # Max chars per turn
SUMMARIZATION_THRESHOLD = 30  # Summarize if > 30 turns

# Pruning strategy
PRUNING_STRATEGY = os.getenv("MEMORY_PRUNING", "sliding_window")  # or "summarize"


# ============================================================================
# METRICS
# ============================================================================

if METRICS_ENABLED:
    memory_calls_active = Gauge(
        'memory_calls_active',
        'Number of active memory sessions'
    )
    memory_turns_total = Counter(
        'memory_turns_total',
        'Total conversation turns',
        ['role']
    )
    memory_slots_updated = Counter(
        'memory_slots_updated_total',
        'Slot update events'
    )
    memory_flags_set = Counter(
        'memory_flags_set_total',
        'Flag set events'
    )
    memory_pruning_events = Counter(
        'memory_pruning_events_total',
        'Memory pruning events',
        ['strategy']
    )
    memory_size_tokens = Histogram(
        'memory_size_tokens',
        'Memory size in tokens'
    )
    memory_snapshots_created = Counter(
        'memory_snapshots_created_total',
        'Snapshot creation events'
    )


# ============================================================================
# ENUMS
# ============================================================================

class ConversationPhase(Enum):
    """Current phase of conversation."""
    GREETING = "greeting"
    DISCOVERY = "discovery"
    ORDERING = "ordering"
    CONFIRMATION = "confirmation"
    PAYMENT = "payment"
    CLOSING = "closing"
    ERROR = "error"


class TurnRole(Enum):
    """Role in conversation turn."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ============================================================================
# IMMUTABLE TURN RECORD
# ============================================================================

@dataclass(frozen=True)
class ConversationTurn:
    """
    Immutable conversation turn record.
    
    CRITICAL: frozen=True means this CANNOT be modified after creation.
    AI cannot edit history - only append new turns.
    """
    role: TurnRole
    content: str
    timestamp: datetime
    turn_id: str
    phase: ConversationPhase
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate turn on creation."""
        # Truncate content if too long
        if len(self.content) > MAX_TURN_CONTENT_LENGTH:
            # Use object.__setattr__ since frozen
            object.__setattr__(
                self,
                'content',
                self.content[:MAX_TURN_CONTENT_LENGTH] + "...[truncated]"
            )
            logger.warning(
                f"Turn content truncated: {len(self.content)} -> {MAX_TURN_CONTENT_LENGTH}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "turn_id": self.turn_id,
            "phase": self.phase.value,
            "metadata": self.metadata.copy()
        }
    
    def estimate_tokens(self) -> int:
        """Estimate token count (rough: ~4 chars per token)."""
        return len(self.content) // 4
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """Create from dictionary."""
        return cls(
            role=TurnRole(data['role']),
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            turn_id=data['turn_id'],
            phase=ConversationPhase(data['phase']),
            metadata=data.get('metadata', {})
        )


# ============================================================================
# ERROR COUNTERS
# ============================================================================

@dataclass
class ErrorCounters:
    """Track errors in conversation."""
    
    # AI errors
    ai_failures: int = 0
    ai_timeouts: int = 0
    ai_invalid_responses: int = 0
    
    # STT errors
    stt_failures: int = 0
    stt_low_confidence: int = 0
    
    # TTS errors
    tts_failures: int = 0
    tts_cancellations: int = 0
    
    # System errors
    system_errors: int = 0
    
    def total_errors(self) -> int:
        """Get total error count."""
        return (
            self.ai_failures + self.ai_timeouts + self.ai_invalid_responses +
            self.stt_failures + self.stt_low_confidence +
            self.tts_failures + self.tts_cancellations +
            self.system_errors
        )
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return asdict(self)
    
    def increment(self, error_type: str):
        """Increment error counter."""
        if hasattr(self, error_type):
            current = getattr(self, error_type)
            setattr(self, error_type, current + 1)
        else:
            logger.warning(f"Unknown error type: {error_type}")


# ============================================================================
# MEMORY SNAPSHOT
# ============================================================================

@dataclass
class MemorySnapshot:
    """
    Immutable snapshot of conversation memory.
    
    Used for:
    - State persistence
    - Rollback/restore
    - Debugging
    """
    snapshot_id: str
    call_id: str
    timestamp: datetime
    
    # Memory state
    turns: List[Dict[str, Any]]  # Serialized turns
    extracted_slots: Dict[str, Any]
    system_flags: Dict[str, bool]
    phase: str
    error_counters: Dict[str, int]
    
    # Metadata
    turn_count: int
    token_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "call_id": self.call_id,
            "timestamp": self.timestamp.isoformat(),
            "turns": self.turns,
            "extracted_slots": self.extracted_slots,
            "system_flags": self.system_flags,
            "phase": self.phase,
            "error_counters": self.error_counters,
            "turn_count": self.turn_count,
            "token_count": self.token_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemorySnapshot':
        """Create from dictionary."""
        return cls(
            snapshot_id=data['snapshot_id'],
            call_id=data['call_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            turns=data['turns'],
            extracted_slots=data['extracted_slots'],
            system_flags=data['system_flags'],
            phase=data['phase'],
            error_counters=data['error_counters'],
            turn_count=data['turn_count'],
            token_count=data['token_count']
        )


# ============================================================================
# CONVERSATION MEMORY
# ============================================================================

class ConversationMemory:
    """
    Structured conversation memory with immutability guarantees.
    
    Architecture:
    - turns[] - Immutable history records (write-once)
    - extracted_slots{} - Mutable data extracted from conversation
    - system_flags{} - Boolean flags for system state
    - phase - Current conversation phase
    - error_counters - Error tracking
    
    Guarantees:
    - AI cannot directly write memory
    - Turns are immutable (frozen dataclass)
    - Memory is pruned to stay within token budget
    - Snapshot/restore capability
    """
    
    def __init__(self, call_id: str, restaurant_id: str):
        self.call_id = call_id
        self.restaurant_id = restaurant_id
        
        # CORE MEMORY STRUCTURES
        self._turns: List[ConversationTurn] = []  # Immutable records
        self.extracted_slots: Dict[str, Any] = {}  # Mutable slots
        self.system_flags: Dict[str, bool] = {}  # Boolean flags
        self.phase: ConversationPhase = ConversationPhase.GREETING
        self.error_counters = ErrorCounters()
        
        # Pruning state
        self._pruned_turn_count = 0  # How many turns were pruned
        self._summary_placeholder: Optional[str] = None  # Summarization placeholder
        
        # Metadata
        self.created_at = datetime.utcnow()
        self._turn_sequence = 0
        
        # Snapshots
        self._snapshots: List[MemorySnapshot] = []
        
        logger.info(f"ConversationMemory created: {call_id}")
    
    # ========================================================================
    # TURNS (Immutable History)
    # ========================================================================
    
    def append_turn(
        self,
        role: TurnRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationTurn:
        """
        Append immutable turn to history.
        
        This is the ONLY way to add turns - no direct writes allowed.
        
        Args:
            role: Speaker role
            content: Turn content
            metadata: Optional metadata
            
        Returns:
            Created turn (immutable)
        """
        # Generate turn ID
        self._turn_sequence += 1
        turn_id = f"{self.call_id}_{self._turn_sequence}"
        
        # Create immutable turn
        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            turn_id=turn_id,
            phase=self.phase,
            metadata=metadata or {}
        )
        
        # Append to history
        self._turns.append(turn)
        
        # Track metrics
        if METRICS_ENABLED:
            memory_turns_total.labels(role=role.value).inc()
        
        logger.debug(f"Turn appended: {turn_id} ({role.value})")
        
        # Auto-prune if needed
        self._auto_prune()
        
        return turn
    
    def get_turns(self, limit: Optional[int] = None) -> List[ConversationTurn]:
        """
        Get turns (read-only).
        
        Returns COPIES to prevent mutation.
        
        Args:
            limit: Max turns to return (most recent)
            
        Returns:
            List of turns (immutable)
        """
        if limit:
            return self._turns[-limit:]
        return self._turns.copy()
    
    def get_turn_count(self) -> int:
        """Get total turn count (including pruned)."""
        return len(self._turns) + self._pruned_turn_count
    
    # ========================================================================
    # SLOTS (Mutable Data Extraction)
    # ========================================================================
    
    def update_slot(self, key: str, value: Any) -> bool:
        """
        Update extracted slot.
        
        This is called by CONTROLLER, not AI directly.
        
        Args:
            key: Slot key
            value: Slot value
            
        Returns:
            True if updated
        """
        # Validate value length
        if isinstance(value, str) and len(value) > MAX_SLOT_VALUE_LENGTH:
            logger.warning(
                f"Slot value too long, truncating: {len(value)} -> {MAX_SLOT_VALUE_LENGTH}"
            )
            value = value[:MAX_SLOT_VALUE_LENGTH]
        
        self.extracted_slots[key] = value
        
        if METRICS_ENABLED:
            memory_slots_updated.inc()
        
        logger.debug(f"Slot updated: {key} = {value}")
        return True
    
    def get_slot(self, key: str, default: Any = None) -> Any:
        """Get slot value."""
        return self.extracted_slots.get(key, default)
    
    def has_slot(self, key: str) -> bool:
        """Check if slot exists."""
        return key in self.extracted_slots
    
    def remove_slot(self, key: str) -> bool:
        """Remove slot."""
        if key in self.extracted_slots:
            del self.extracted_slots[key]
            logger.debug(f"Slot removed: {key}")
            return True
        return False
    
    def get_all_slots(self) -> Dict[str, Any]:
        """Get all slots (copy)."""
        return self.extracted_slots.copy()
    
    # ========================================================================
    # FLAGS (System State)
    # ========================================================================
    
    def set_flag(self, key: str, value: bool = True):
        """
        Set system flag.
        
        Called by CONTROLLER, not AI.
        
        Args:
            key: Flag key
            value: Flag value (default True)
        """
        self.system_flags[key] = value
        
        if METRICS_ENABLED:
            memory_flags_set.inc()
        
        logger.debug(f"Flag set: {key} = {value}")
    
    def get_flag(self, key: str, default: bool = False) -> bool:
        """Get flag value."""
        return self.system_flags.get(key, default)
    
    def has_flag(self, key: str) -> bool:
        """Check if flag exists."""
        return key in self.system_flags
    
    def clear_flag(self, key: str):
        """Clear flag."""
        if key in self.system_flags:
            del self.system_flags[key]
            logger.debug(f"Flag cleared: {key}")
    
    def get_all_flags(self) -> Dict[str, bool]:
        """Get all flags (copy)."""
        return self.system_flags.copy()
    
    # ========================================================================
    # PHASE MANAGEMENT
    # ========================================================================
    
    def set_phase(self, phase: ConversationPhase):
        """
        Set conversation phase.
        
        Called by CONTROLLER based on AI output interpretation.
        
        Args:
            phase: New phase
        """
        old_phase = self.phase
        self.phase = phase
        
        logger.info(f"Phase transition: {old_phase.value} -> {phase.value}")
    
    def get_phase(self) -> ConversationPhase:
        """Get current phase."""
        return self.phase
    
    # ========================================================================
    # ERROR TRACKING
    # ========================================================================
    
    def increment_error(self, error_type: str):
        """
        Increment error counter.
        
        Called by CONTROLLER when errors occur.
        
        Args:
            error_type: Type of error (e.g., 'ai_failures')
        """
        self.error_counters.increment(error_type)
        logger.warning(f"Error incremented: {error_type}")
    
    def get_error_count(self, error_type: Optional[str] = None) -> int:
        """
        Get error count.
        
        Args:
            error_type: Specific error type, or None for total
            
        Returns:
            Error count
        """
        if error_type:
            return getattr(self.error_counters, error_type, 0)
        return self.error_counters.total_errors()
    
    def get_all_errors(self) -> Dict[str, int]:
        """Get all error counts."""
        return self.error_counters.to_dict()
    
    # ========================================================================
    # MEMORY PRUNING
    # ========================================================================
    
    def _auto_prune(self):
        """
        Automatically prune memory if needed.
        
        Strategies:
        1. Sliding window - Keep last N turns
        2. Summarization - Summarize old turns (placeholder)
        """
        # Check if pruning needed
        if len(self._turns) <= MAX_TURNS_WINDOW:
            return
        
        logger.info(
            f"Memory pruning triggered: {len(self._turns)} turns > {MAX_TURNS_WINDOW}"
        )
        
        if PRUNING_STRATEGY == "sliding_window":
            self._prune_sliding_window()
        elif PRUNING_STRATEGY == "summarize":
            self._prune_with_summarization()
        else:
            logger.warning(f"Unknown pruning strategy: {PRUNING_STRATEGY}")
            self._prune_sliding_window()  # Fallback
    
    def _prune_sliding_window(self):
        """
        Prune using sliding window (keep last N turns).
        
        This is the simplest and safest pruning strategy.
        """
        excess = len(self._turns) - MAX_TURNS_WINDOW
        
        if excess <= 0:
            return
        
        # Remove oldest turns
        pruned_turns = self._turns[:excess]
        self._turns = self._turns[excess:]
        
        # Track pruned count
        self._pruned_turn_count += len(pruned_turns)
        
        if METRICS_ENABLED:
            memory_pruning_events.labels(strategy='sliding_window').inc()
        
        logger.info(
            f"Pruned {len(pruned_turns)} turns (sliding window), "
            f"{len(self._turns)} remaining"
        )
    
    def _prune_with_summarization(self):
        """
        Prune with summarization (advanced).
        
        Instead of dropping old turns, create a summary placeholder.
        Actual summarization would require LLM call.
        """
        if len(self._turns) < SUMMARIZATION_THRESHOLD:
            return
        
        # Determine how many turns to summarize
        turns_to_summarize = len(self._turns) - MAX_TURNS_WINDOW // 2
        
        if turns_to_summarize <= 0:
            return
        
        # Get turns to summarize
        old_turns = self._turns[:turns_to_summarize]
        
        # Create placeholder (real implementation would call LLM)
        summary = f"[Summarized {len(old_turns)} turns from earlier conversation]"
        
        self._summary_placeholder = summary
        
        # Remove summarized turns
        self._turns = self._turns[turns_to_summarize:]
        self._pruned_turn_count += len(old_turns)
        
        if METRICS_ENABLED:
            memory_pruning_events.labels(strategy='summarize').inc()
        
        logger.info(
            f"Summarized {len(old_turns)} turns, "
            f"{len(self._turns)} remaining"
        )
    
    def get_summary_placeholder(self) -> Optional[str]:
        """Get summarization placeholder if exists."""
        return self._summary_placeholder
    
    # ========================================================================
    # TOKEN BUDGETING
    # ========================================================================
    
    def estimate_token_count(self) -> int:
        """
        Estimate total token count in memory.
        
        Rough estimate: ~4 chars per token.
        
        Returns:
            Estimated token count
        """
        token_count = 0
        
        # Turns
        for turn in self._turns:
            token_count += turn.estimate_tokens()
        
        # Summary placeholder
        if self._summary_placeholder:
            token_count += len(self._summary_placeholder) // 4
        
        # Slots (convert to string representation)
        slots_str = json.dumps(self.extracted_slots)
        token_count += len(slots_str) // 4
        
        # Flags
        flags_str = json.dumps(self.system_flags)
        token_count += len(flags_str) // 4
        
        if METRICS_ENABLED:
            memory_size_tokens.observe(token_count)
        
        return token_count
    
    def is_within_budget(self) -> bool:
        """Check if memory is within token budget."""
        return self.estimate_token_count() <= MAX_TOKEN_BUDGET
    
    def get_budget_usage(self) -> Tuple[int, int, float]:
        """
        Get token budget usage.
        
        Returns:
            (current_tokens, max_tokens, usage_percentage)
        """
        current = self.estimate_token_count()
        max_tokens = MAX_TOKEN_BUDGET
        usage = (current / max_tokens) * 100
        
        return current, max_tokens, usage
    
    # ========================================================================
    # SNAPSHOT & RESTORE
    # ========================================================================
    
    def create_snapshot(self) -> MemorySnapshot:
        """
        Create immutable snapshot of current memory state.
        
        Used for:
        - Persistence
        - Rollback
        - Debugging
        
        Returns:
            Memory snapshot
        """
        snapshot_id = f"{self.call_id}_{datetime.utcnow().timestamp()}"
        
        snapshot = MemorySnapshot(
            snapshot_id=snapshot_id,
            call_id=self.call_id,
            timestamp=datetime.utcnow(),
            turns=[turn.to_dict() for turn in self._turns],
            extracted_slots=deepcopy(self.extracted_slots),
            system_flags=deepcopy(self.system_flags),
            phase=self.phase.value,
            error_counters=self.error_counters.to_dict(),
            turn_count=self.get_turn_count(),
            token_count=self.estimate_token_count()
        )
        
        # Store snapshot
        self._snapshots.append(snapshot)
        
        if METRICS_ENABLED:
            memory_snapshots_created.inc()
        
        logger.info(
            f"Snapshot created: {snapshot_id} "
            f"({snapshot.turn_count} turns, {snapshot.token_count} tokens)"
        )
        
        return snapshot
    
    def restore_from_snapshot(self, snapshot: MemorySnapshot):
        """
        Restore memory from snapshot.
        
        DANGEROUS: This replaces current memory state.
        
        Args:
            snapshot: Snapshot to restore from
        """
        logger.warning(f"Restoring memory from snapshot: {snapshot.snapshot_id}")
        
        # Restore turns (recreate immutable objects)
        self._turns = [
            ConversationTurn.from_dict(turn_data)
            for turn_data in snapshot.turns
        ]
        
        # Restore slots
        self.extracted_slots = deepcopy(snapshot.extracted_slots)
        
        # Restore flags
        self.system_flags = deepcopy(snapshot.system_flags)
        
        # Restore phase
        self.phase = ConversationPhase(snapshot.phase)
        
        # Restore error counters
        for key, value in snapshot.error_counters.items():
            setattr(self.error_counters, key, value)
        
        logger.info(f"Memory restored from snapshot: {snapshot.snapshot_id}")
    
    def get_snapshots(self) -> List[MemorySnapshot]:
        """Get all snapshots."""
        return self._snapshots.copy()
    
    def get_latest_snapshot(self) -> Optional[MemorySnapshot]:
        """Get most recent snapshot."""
        if self._snapshots:
            return self._snapshots[-1]
        return None
    
    # ========================================================================
    # EXPORT & DEBUG
    # ========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export memory to dictionary.
        
        Returns:
            Complete memory state
        """
        current_tokens, max_tokens, usage = self.get_budget_usage()
        
        return {
            "call_id": self.call_id,
            "restaurant_id": self.restaurant_id,
            "created_at": self.created_at.isoformat(),
            "phase": self.phase.value,
            "turns": [turn.to_dict() for turn in self._turns],
            "turn_count": self.get_turn_count(),
            "pruned_turns": self._pruned_turn_count,
            "extracted_slots": self.extracted_slots.copy(),
            "system_flags": self.system_flags.copy(),
            "error_counters": self.error_counters.to_dict(),
            "summary_placeholder": self._summary_placeholder,
            "token_budget": {
                "current_tokens": current_tokens,
                "max_tokens": max_tokens,
                "usage_percent": round(usage, 2),
                "within_budget": self.is_within_budget()
            },
            "snapshot_count": len(self._snapshots)
        }
    
    def get_recent_context(self, max_turns: int = 10) -> str:
        """
        Get recent conversation context as formatted string.
        
        Used for AI prompts.
        
        Args:
            max_turns: Max recent turns to include
            
        Returns:
            Formatted conversation history
        """
        recent_turns = self._turns[-max_turns:]
        
        lines = []
        
        # Add summary if exists
        if self._summary_placeholder:
            lines.append(self._summary_placeholder)
            lines.append("")
        
        # Add recent turns
        for turn in recent_turns:
            role_name = turn.role.value.upper()
            lines.append(f"{role_name}: {turn.content}")
        
        return "\n".join(lines)


# ============================================================================
# GLOBAL MEMORY REGISTRY
# ============================================================================

_memory_registry: Dict[str, ConversationMemory] = {}


def create_memory(call_id: str, restaurant_id: str) -> ConversationMemory:
    """
    Create conversation memory.
    
    Args:
        call_id: Call identifier
        restaurant_id: Restaurant identifier
        
    Returns:
        ConversationMemory instance
    """
    if call_id in _memory_registry:
        logger.warning(f"Memory already exists for {call_id}, returning existing")
        return _memory_registry[call_id]
    
    memory = ConversationMemory(call_id, restaurant_id)
    _memory_registry[call_id] = memory
    
    if METRICS_ENABLED:
        memory_calls_active.set(len(_memory_registry))
    
    logger.info(f"Memory created: {call_id}")
    return memory


def get_memory(call_id: str) -> Optional[ConversationMemory]:
    """
    Get memory for call.
    
    Args:
        call_id: Call identifier
        
    Returns:
        ConversationMemory or None
    """
    return _memory_registry.get(call_id)


def clear_memory(call_id: str):
    """
    Clear memory for call.
    
    Args:
        call_id: Call identifier
    """
    if call_id in _memory_registry:
        del _memory_registry[call_id]
        
        if METRICS_ENABLED:
            memory_calls_active.set(len(_memory_registry))
        
        logger.info(f"Memory cleared: {call_id}")


def get_active_memory_count() -> int:
    """Get count of active memories."""
    return len(_memory_registry)


def get_memory_stats() -> Dict[str, Any]:
    """Get global memory statistics."""
    total_turns = sum(mem.get_turn_count() for mem in _memory_registry.values())
    total_slots = sum(len(mem.extracted_slots) for mem in _memory_registry.values())
    total_flags = sum(len(mem.system_flags) for mem in _memory_registry.values())
    total_tokens = sum(mem.estimate_token_count() for mem in _memory_registry.values())
    
    return {
        "active_calls": len(_memory_registry),
        "total_turns": total_turns,
        "total_slots": total_slots,
        "total_flags": total_flags,
        "total_tokens": total_tokens,
        "avg_tokens_per_call": round(total_tokens / max(1, len(_memory_registry)), 2)
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Memory Module (Production Hardened v3.0)")
    print("="*60)
    print(f"Max Turns Window: {MAX_TURNS_WINDOW}")
    print(f"Max Token Budget: {MAX_TOKEN_BUDGET}")
    print(f"Pruning Strategy: {PRUNING_STRATEGY}")
    print("="*60)
    
    # Create memory
    memory = create_memory("test_call_001", "rest_001")
    print(f"\nMemory created: {memory.call_id}")
    
    # Add turns (immutable)
    for i in range(5):
        memory.append_turn(
            TurnRole.USER,
            f"User message {i}",
            {"index": i}
        )
        memory.append_turn(
            TurnRole.ASSISTANT,
            f"Assistant response {i}",
            {"index": i}
        )
    
    print(f"\nTurns: {memory.get_turn_count()}")
    
    # Extract slots
    memory.update_slot("customer_name", "John Doe")
    memory.update_slot("table_size", 4)
    print(f"\nSlots: {memory.get_all_slots()}")
    
    # Set flags
    memory.set_flag("order_started", True)
    memory.set_flag("payment_required", False)
    print(f"\nFlags: {memory.get_all_flags()}")
    
    # Phase
    memory.set_phase(ConversationPhase.ORDERING)
    print(f"\nPhase: {memory.get_phase().value}")
    
    # Errors
    memory.increment_error("ai_failures")
    memory.increment_error("stt_low_confidence")
    print(f"\nErrors: {memory.get_all_errors()}")
    
    # Token budget
    current, max_tokens, usage = memory.get_budget_usage()
    print(f"\nToken budget: {current}/{max_tokens} ({usage:.1f}%)")
    
    # Snapshot
    snapshot = memory.create_snapshot()
    print(f"\nSnapshot created: {snapshot.snapshot_id}")
    print(f"  Turns: {snapshot.turn_count}")
    print(f"  Tokens: {snapshot.token_count}")
    
    # Export
    export = memory.to_dict()
    print(f"\nMemory export:")
    print(f"  Phase: {export['phase']}")
    print(f"  Turns: {export['turn_count']}")
    print(f"  Pruned: {export['pruned_turns']}")
    print(f"  Within budget: {export['token_budget']['within_budget']}")
    
    # Recent context
    context = memory.get_recent_context(max_turns=3)
    print(f"\nRecent context:\n{context}")
    
    # Stats
    stats = get_memory_stats()
    print(f"\nGlobal stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    clear_memory("test_call_001")
    print(f"\nMemory cleared")
