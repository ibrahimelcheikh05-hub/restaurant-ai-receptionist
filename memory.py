"""
In-Call Memory Engine (Production)
===================================
Structured memory management with explicit lifecycle.
Per-call isolation, history trimming, memory caps, summarization hooks.
Only main.py may write to memory - strict ownership enforcement.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum


logger = logging.getLogger(__name__)


class MemoryState(Enum):
    """Memory lifecycle states."""
    CREATED = "created"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CLOSED = "closed"


@dataclass
class ConversationTurn:
    """Single conversation turn."""
    role: str  # "user" or "assistant"
    content: str
    intent: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemorySnapshot:
    """Point-in-time memory snapshot."""
    call_id: str
    restaurant_id: str
    state: str
    facts: Dict[str, Any]
    flags: Dict[str, bool]
    slots: Dict[str, Any]
    turns: List[ConversationTurn]
    error_count: int
    created_at: str
    updated_at: str
    turn_count: int


class CallMemory:
    """
    Structured in-call memory with lifecycle management.
    Supports facts, flags, slots, conversation turns, and error tracking.
    """
    
    def __init__(self, call_id: str, restaurant_id: str):
        """
        Initialize call memory.
        
        Args:
            call_id: Unique call identifier
            restaurant_id: Restaurant identifier
        """
        self.call_id = call_id
        self.restaurant_id = restaurant_id
        self.state = MemoryState.CREATED
        
        # Structured memory components
        self.facts: Dict[str, Any] = {}  # Immutable facts (customer name, etc.)
        self.flags: Dict[str, bool] = {}  # Boolean flags (upsell_offered, etc.)
        self.slots: Dict[str, Any] = {}  # Mutable slots (current_item, etc.)
        self.turns: List[ConversationTurn] = []  # Conversation history
        self.errors: List[str] = []  # Error log
        
        # Menu snapshot (set once, never modified)
        self._menu_snapshot: Optional[Dict[str, Any]] = None
        self._menu_locked = False
        
        # Lifecycle tracking
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        # Caps
        self.max_turns = 200
        self.max_errors = 50
        self.max_facts = 100
        self.max_flags = 50
        self.max_slots = 50
        
        logger.info(f"Memory created for call {call_id}")
    
    # ========================================================================
    # LIFECYCLE MANAGEMENT
    # ========================================================================
    
    def activate(self):
        """Transition to active state."""
        if self.state == MemoryState.CREATED:
            self.state = MemoryState.ACTIVE
            self._touch()
            logger.info(f"Memory activated for {self.call_id}")
    
    def suspend(self):
        """Suspend memory (pause call)."""
        if self.state == MemoryState.ACTIVE:
            self.state = MemoryState.SUSPENDED
            self._touch()
            logger.info(f"Memory suspended for {self.call_id}")
    
    def close(self):
        """Close memory (end call)."""
        if self.state != MemoryState.CLOSED:
            self.state = MemoryState.CLOSED
            self._touch()
            logger.info(
                f"Memory closed for {self.call_id}: "
                f"{len(self.turns)} turns, {len(self.errors)} errors"
            )
    
    def is_active(self) -> bool:
        """Check if memory is active."""
        return self.state == MemoryState.ACTIVE
    
    def is_closed(self) -> bool:
        """Check if memory is closed."""
        return self.state == MemoryState.CLOSED
    
    def _touch(self):
        """Update timestamp."""
        self.updated_at = datetime.utcnow()
    
    # ========================================================================
    # FACTS (Immutable)
    # ========================================================================
    
    def set_fact(self, key: str, value: Any) -> bool:
        """
        Set immutable fact (write-once).
        
        Args:
            key: Fact key
            value: Fact value
            
        Returns:
            True if set, False if already exists
        """
        if key in self.facts:
            logger.warning(f"Fact '{key}' already set, ignoring update")
            return False
        
        if len(self.facts) >= self.max_facts:
            logger.warning(f"Max facts reached ({self.max_facts})")
            return False
        
        self.facts[key] = value
        self._touch()
        logger.debug(f"Fact set: {key}={value}")
        return True
    
    def get_fact(self, key: str, default: Any = None) -> Any:
        """Get fact value."""
        return self.facts.get(key, default)
    
    def has_fact(self, key: str) -> bool:
        """Check if fact exists."""
        return key in self.facts
    
    def get_all_facts(self) -> Dict[str, Any]:
        """Get all facts."""
        return self.facts.copy()
    
    # ========================================================================
    # FLAGS (Boolean state)
    # ========================================================================
    
    def set_flag(self, key: str, value: bool = True):
        """
        Set boolean flag.
        
        Args:
            key: Flag key
            value: Flag value (default: True)
        """
        if len(self.flags) >= self.max_flags and key not in self.flags:
            logger.warning(f"Max flags reached ({self.max_flags})")
            return
        
        self.flags[key] = value
        self._touch()
        logger.debug(f"Flag set: {key}={value}")
    
    def get_flag(self, key: str, default: bool = False) -> bool:
        """Get flag value."""
        return self.flags.get(key, default)
    
    def toggle_flag(self, key: str):
        """Toggle flag value."""
        self.flags[key] = not self.flags.get(key, False)
        self._touch()
    
    def get_all_flags(self) -> Dict[str, bool]:
        """Get all flags."""
        return self.flags.copy()
    
    # ========================================================================
    # SLOTS (Mutable state)
    # ========================================================================
    
    def set_slot(self, key: str, value: Any):
        """
        Set mutable slot.
        
        Args:
            key: Slot key
            value: Slot value
        """
        if len(self.slots) >= self.max_slots and key not in self.slots:
            logger.warning(f"Max slots reached ({self.max_slots})")
            return
        
        self.slots[key] = value
        self._touch()
        logger.debug(f"Slot set: {key}={value}")
    
    def get_slot(self, key: str, default: Any = None) -> Any:
        """Get slot value."""
        return self.slots.get(key, default)
    
    def clear_slot(self, key: str):
        """Clear slot value."""
        if key in self.slots:
            del self.slots[key]
            self._touch()
    
    def get_all_slots(self) -> Dict[str, Any]:
        """Get all slots."""
        return self.slots.copy()
    
    # ========================================================================
    # CONVERSATION TURNS
    # ========================================================================
    
    def add_conversation_turn(
        self,
        role: str,
        content: str,
        intent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add conversation turn.
        
        Args:
            role: "user" or "assistant"
            content: Turn content
            intent: Intent classification (optional)
            metadata: Additional metadata (optional)
        """
        if role not in ["user", "assistant", "system"]:
            logger.warning(f"Invalid role: {role}")
            return
        
        turn = ConversationTurn(
            role=role,
            content=content,
            intent=intent,
            metadata=metadata or {}
        )
        
        self.turns.append(turn)
        self._touch()
        
        # Auto-trim if needed
        if len(self.turns) > self.max_turns:
            self._trim_turns()
        
        logger.debug(f"Turn added: {role} ({len(content)} chars)")
    
    def get_conversation_history(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Args:
            last_n: Return only last N turns (optional)
            
        Returns:
            List of turn dictionaries
        """
        turns = self.turns[-last_n:] if last_n else self.turns
        return [asdict(turn) for turn in turns]
    
    def get_turn_count(self) -> int:
        """Get total turn count."""
        return len(self.turns)
    
    def get_user_turns(self) -> List[Dict[str, Any]]:
        """Get only user turns."""
        return [asdict(t) for t in self.turns if t.role == "user"]
    
    def get_assistant_turns(self) -> List[Dict[str, Any]]:
        """Get only assistant turns."""
        return [asdict(t) for t in self.turns if t.role == "assistant"]
    
    def _trim_turns(self):
        """Trim conversation history (keep recent turns)."""
        keep_count = int(self.max_turns * 0.75)  # Keep 75%
        removed_count = len(self.turns) - keep_count
        
        if removed_count > 0:
            self.turns = self.turns[-keep_count:]
            logger.info(f"Trimmed {removed_count} old turns from {self.call_id}")
    
    def get_summary_hook_data(self) -> Dict[str, Any]:
        """
        Get data for conversation summarization.
        To be used by external summarization service.
        
        Returns:
            Dictionary with summarizable data
        """
        return {
            "call_id": self.call_id,
            "turn_count": len(self.turns),
            "user_turns": [t.content for t in self.turns if t.role == "user"],
            "assistant_turns": [t.content for t in self.turns if t.role == "assistant"],
            "intents": [t.intent for t in self.turns if t.intent],
            "facts": self.facts,
            "flags": self.flags
        }
    
    # ========================================================================
    # ERROR TRACKING
    # ========================================================================
    
    def log_error(self, error_msg: str):
        """
        Log error.
        
        Args:
            error_msg: Error message
        """
        if len(self.errors) >= self.max_errors:
            # Trim oldest errors
            self.errors = self.errors[-int(self.max_errors * 0.75):]
        
        timestamp = datetime.utcnow().isoformat()
        self.errors.append(f"[{timestamp}] {error_msg}")
        self._touch()
        
        logger.warning(f"Error logged for {self.call_id}: {error_msg}")
    
    def get_error_count(self) -> int:
        """Get error count."""
        return len(self.errors)
    
    def get_errors(self) -> List[str]:
        """Get all errors."""
        return self.errors.copy()
    
    def clear_errors(self):
        """Clear error log."""
        self.errors.clear()
        self._touch()
    
    # ========================================================================
    # MENU SNAPSHOT (Write-once)
    # ========================================================================
    
    def set_menu_snapshot(self, menu: Dict[str, Any]):
        """
        Set menu snapshot (write-once).
        
        Args:
            menu: Menu data
        """
        if self._menu_locked:
            logger.warning(f"Menu already set for {self.call_id}, ignoring update")
            return
        
        self._menu_snapshot = menu
        self._menu_locked = True
        self._touch()
        logger.info(f"Menu snapshot set for {self.call_id}")
    
    def get_menu_snapshot(self) -> Optional[Dict[str, Any]]:
        """Get menu snapshot."""
        return self._menu_snapshot
    
    def has_menu(self) -> bool:
        """Check if menu is set."""
        return self._menu_snapshot is not None
    
    # ========================================================================
    # STATE MANAGEMENT
    # ========================================================================
    
    def set_state(self, state_key: str, state_value: Any = True):
        """
        Set arbitrary state (convenience wrapper for slots).
        
        Args:
            state_key: State key
            state_value: State value
        """
        self.set_slot(state_key, state_value)
    
    def get_state(self, state_key: str, default: Any = None) -> Any:
        """Get arbitrary state."""
        return self.get_slot(state_key, default)
    
    # ========================================================================
    # SNAPSHOT
    # ========================================================================
    
    def snapshot(self) -> MemorySnapshot:
        """
        Create point-in-time snapshot.
        
        Returns:
            MemorySnapshot instance
        """
        return MemorySnapshot(
            call_id=self.call_id,
            restaurant_id=self.restaurant_id,
            state=self.state.value,
            facts=self.facts.copy(),
            flags=self.flags.copy(),
            slots=self.slots.copy(),
            turns=self.turns.copy(),
            error_count=len(self.errors),
            created_at=self.created_at.isoformat(),
            updated_at=self.updated_at.isoformat(),
            turn_count=len(self.turns)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self.snapshot())


# ============================================================================
# GLOBAL MEMORY STORE (Per-Call Isolation)
# ============================================================================

_memory_store: Dict[str, CallMemory] = {}


def create_call_memory(call_id: str, restaurant_id: str) -> CallMemory:
    """
    Create new call memory.
    
    Args:
        call_id: Call identifier
        restaurant_id: Restaurant identifier
        
    Returns:
        CallMemory instance
    """
    if call_id in _memory_store:
        logger.warning(f"Memory already exists for {call_id}, returning existing")
        return _memory_store[call_id]
    
    memory = CallMemory(call_id, restaurant_id)
    memory.activate()
    _memory_store[call_id] = memory
    
    logger.info(f"Created memory for call {call_id}")
    return memory


def get_memory(call_id: str) -> Optional[CallMemory]:
    """
    Get call memory.
    
    Args:
        call_id: Call identifier
        
    Returns:
        CallMemory instance or None
    """
    return _memory_store.get(call_id)


def clear_memory(call_id: str):
    """
    Clear call memory (cleanup).
    
    Args:
        call_id: Call identifier
    """
    memory = _memory_store.pop(call_id, None)
    
    if memory:
        memory.close()
        logger.info(f"Cleared memory for call {call_id}")


def get_active_memories() -> List[str]:
    """
    Get list of active memory call IDs.
    
    Returns:
        List of call IDs
    """
    return list(_memory_store.keys())


def get_memory_count() -> int:
    """Get count of active memories."""
    return len(_memory_store)


def clear_all_memories():
    """Clear all memories (for testing/cleanup)."""
    count = len(_memory_store)
    
    for memory in _memory_store.values():
        memory.close()
    
    _memory_store.clear()
    logger.info(f"Cleared all {count} memories")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("In-Call Memory Engine (Production)")
    print("=" * 50)
    
    # Create memory
    call_id = "test_call_123"
    memory = create_call_memory(call_id, "rest_001")
    
    print(f"\nMemory created: {call_id}")
    print(f"State: {memory.state.value}")
    
    # Set facts
    memory.set_fact("customer_name", "John")
    memory.set_fact("customer_phone", "+1234567890")
    print(f"\nFacts: {memory.get_all_facts()}")
    
    # Set flags
    memory.set_flag("upsell_offered", True)
    memory.set_flag("order_confirmed", False)
    print(f"Flags: {memory.get_all_flags()}")
    
    # Set slots
    memory.set_slot("current_item", "Large Pizza")
    memory.set_slot("order_total", 15.99)
    print(f"Slots: {memory.get_all_slots()}")
    
    # Add turns
    memory.add_conversation_turn("user", "I want a large pizza", "order")
    memory.add_conversation_turn("assistant", "Great! What toppings?", "clarify")
    memory.add_conversation_turn("user", "Pepperoni and mushrooms", "order_detail")
    
    print(f"\nConversation: {memory.get_turn_count()} turns")
    
    # Snapshot
    snapshot = memory.snapshot()
    print(f"\nSnapshot created:")
    print(f"  Turn count: {snapshot.turn_count}")
    print(f"  Error count: {snapshot.error_count}")
    
    # Cleanup
    clear_memory(call_id)
    print(f"\nMemory cleared: {call_id}")
    print(f"Active memories: {get_memory_count()}")
    
    print("\n" + "=" * 50)
    print("Production memory engine ready")
