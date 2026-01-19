"""
In-Call Memory Engine
=====================
Manages temporary state for active voice calls.
Pure in-memory storage - no persistence, no database.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from threading import Lock


class CallMemory:
    """
    In-memory storage for a single call's state.
    Stores all temporary data needed during an active conversation.
    """
    
    def __init__(self, call_id: str):
        """
        Initialize memory for a specific call.
        
        Args:
            call_id: Unique identifier for the call
        """
        self.call_id = call_id
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        # Core memory fields
        self.detected_language: Optional[str] = None
        self.menu_snapshot: List[Dict[str, Any]] = []
        self.current_order: Dict[str, Any] = {
            "items": [],
            "total": 0.0,
            "tax": 0.0,
            "subtotal": 0.0
        }
        self.last_intent: Optional[str] = None
        self.conversation_context: List[Dict[str, Any]] = []
        
        # Additional useful fields
        self.customer_phone: Optional[str] = None
        self.customer_name: Optional[str] = None
        self.restaurant_id: Optional[str] = None
        self.call_start_time: datetime = datetime.utcnow()
        
        # State tracking
        self.state: str = "initialized"  # initialized, active, completed, failed
        self.turn_count: int = 0
        
        # Lock for thread-safe updates
        self._lock = Lock()
    
    def update_language(self, language: str) -> None:
        """
        Update detected language.
        
        Args:
            language: Language code (e.g., 'en', 'es', 'fr')
        """
        with self._lock:
            self.detected_language = language
            self._touch()
    
    def set_menu_snapshot(self, menu: List[Dict[str, Any]]) -> None:
        """
        Store menu snapshot for this call.
        
        Args:
            menu: List of menu item dictionaries
        """
        with self._lock:
            self.menu_snapshot = menu.copy()
            self._touch()
    
    def add_order_item(
        self, 
        item_id: str, 
        name: str,
        quantity: int = 1,
        price: float = 0.0,
        customizations: Optional[List[str]] = None
    ) -> None:
        """
        Add item to current order.
        
        Args:
            item_id: Unique identifier for menu item
            name: Item name
            quantity: Number of items
            price: Price per item
            customizations: List of customizations/modifications
        """
        with self._lock:
            order_item = {
                "item_id": item_id,
                "name": name,
                "quantity": quantity,
                "price": price,
                "customizations": customizations or [],
                "added_at": datetime.utcnow().isoformat()
            }
            self.current_order["items"].append(order_item)
            self._recalculate_totals()
            self._touch()
    
    def remove_order_item(self, item_id: str) -> bool:
        """
        Remove item from current order.
        
        Args:
            item_id: ID of item to remove
            
        Returns:
            True if item was removed, False if not found
        """
        with self._lock:
            original_length = len(self.current_order["items"])
            self.current_order["items"] = [
                item for item in self.current_order["items"]
                if item["item_id"] != item_id
            ]
            removed = len(self.current_order["items"]) < original_length
            if removed:
                self._recalculate_totals()
                self._touch()
            return removed
    
    def update_order_item_quantity(
        self, 
        item_id: str, 
        quantity: int
    ) -> bool:
        """
        Update quantity for an item in the order.
        
        Args:
            item_id: ID of item to update
            quantity: New quantity
            
        Returns:
            True if updated, False if item not found
        """
        with self._lock:
            for item in self.current_order["items"]:
                if item["item_id"] == item_id:
                    item["quantity"] = quantity
                    self._recalculate_totals()
                    self._touch()
                    return True
            return False
    
    def clear_order(self) -> None:
        """Clear all items from current order."""
        with self._lock:
            self.current_order = {
                "items": [],
                "total": 0.0,
                "tax": 0.0,
                "subtotal": 0.0
            }
            self._touch()
    
    def set_last_intent(self, intent: str) -> None:
        """
        Record the last detected user intent.
        
        Args:
            intent: Intent name (e.g., 'add_item', 'remove_item', 'checkout')
        """
        with self._lock:
            self.last_intent = intent
            self._touch()
    
    def add_conversation_turn(
        self,
        role: str,
        content: str,
        intent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a conversation turn to context.
        
        Args:
            role: Speaker role ('user' or 'assistant')
            content: What was said
            intent: Detected intent for this turn
            metadata: Additional turn metadata
        """
        with self._lock:
            turn = {
                "role": role,
                "content": content,
                "intent": intent,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat()
            }
            self.conversation_context.append(turn)
            self.turn_count += 1
            self._touch()
    
    def get_recent_context(self, num_turns: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent conversation turns.
        
        Args:
            num_turns: Number of recent turns to retrieve
            
        Returns:
            List of recent conversation turns
        """
        with self._lock:
            return self.conversation_context[-num_turns:] if self.conversation_context else []
    
    def set_customer_info(
        self, 
        phone: Optional[str] = None,
        name: Optional[str] = None
    ) -> None:
        """
        Set customer information.
        
        Args:
            phone: Customer phone number
            name: Customer name
        """
        with self._lock:
            if phone:
                self.customer_phone = phone
            if name:
                self.customer_name = name
            self._touch()
    
    def set_restaurant_id(self, restaurant_id: str) -> None:
        """
        Set restaurant ID for this call.
        
        Args:
            restaurant_id: Restaurant identifier
        """
        with self._lock:
            self.restaurant_id = restaurant_id
            self._touch()
    
    def set_state(self, state: str) -> None:
        """
        Update call state.
        
        Args:
            state: New state (initialized, active, completed, failed)
        """
        with self._lock:
            self.state = state
            self._touch()
    
    def get_order_summary(self) -> Dict[str, Any]:
        """
        Get current order summary.
        
        Returns:
            Dictionary with order details and totals
        """
        with self._lock:
            return {
                "items": self.current_order["items"].copy(),
                "subtotal": self.current_order["subtotal"],
                "tax": self.current_order["tax"],
                "total": self.current_order["total"],
                "item_count": len(self.current_order["items"])
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert memory to dictionary for serialization.
        
        Returns:
            Dictionary representation of call memory
        """
        with self._lock:
            return {
                "call_id": self.call_id,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
                "detected_language": self.detected_language,
                "menu_snapshot": self.menu_snapshot.copy(),
                "current_order": {
                    "items": self.current_order["items"].copy(),
                    "subtotal": self.current_order["subtotal"],
                    "tax": self.current_order["tax"],
                    "total": self.current_order["total"]
                },
                "last_intent": self.last_intent,
                "conversation_context": self.conversation_context.copy(),
                "customer_phone": self.customer_phone,
                "customer_name": self.customer_name,
                "restaurant_id": self.restaurant_id,
                "state": self.state,
                "turn_count": self.turn_count,
                "call_start_time": self.call_start_time.isoformat()
            }
    
    def _recalculate_totals(self) -> None:
        """Recalculate order totals based on items (internal use)."""
        subtotal = sum(
            item["price"] * item["quantity"]
            for item in self.current_order["items"]
        )
        tax = subtotal * 0.08  # Default 8% tax, should come from restaurant settings
        total = subtotal + tax
        
        self.current_order["subtotal"] = round(subtotal, 2)
        self.current_order["tax"] = round(tax, 2)
        self.current_order["total"] = round(total, 2)
    
    def _touch(self) -> None:
        """Update the last modified timestamp (internal use)."""
        self.updated_at = datetime.utcnow()


class MemoryStore:
    """
    Global store for managing all active call memories.
    Provides per-call isolation with thread-safe access.
    """
    
    def __init__(self):
        """Initialize the memory store."""
        self._memories: Dict[str, CallMemory] = {}
        self._lock = Lock()
    
    def create_call_memory(self, call_id: str) -> CallMemory:
        """
        Create a new call memory instance.
        
        Args:
            call_id: Unique identifier for the call
            
        Returns:
            New CallMemory instance
            
        Raises:
            ValueError: If call_id already exists
        """
        with self._lock:
            if call_id in self._memories:
                raise ValueError(f"Memory for call_id '{call_id}' already exists")
            
            memory = CallMemory(call_id)
            self._memories[call_id] = memory
            return memory
    
    def get_memory(self, call_id: str) -> Optional[CallMemory]:
        """
        Retrieve memory for a specific call.
        
        Args:
            call_id: Call identifier
            
        Returns:
            CallMemory instance or None if not found
        """
        with self._lock:
            return self._memories.get(call_id)
    
    def clear_memory(self, call_id: str) -> bool:
        """
        Clear and remove memory for a call.
        
        Args:
            call_id: Call identifier
            
        Returns:
            True if memory was cleared, False if not found
        """
        with self._lock:
            if call_id in self._memories:
                del self._memories[call_id]
                return True
            return False
    
    def get_or_create(self, call_id: str) -> CallMemory:
        """
        Get existing memory or create new one if doesn't exist.
        
        Args:
            call_id: Call identifier
            
        Returns:
            CallMemory instance (existing or new)
        """
        with self._lock:
            if call_id not in self._memories:
                self._memories[call_id] = CallMemory(call_id)
            return self._memories[call_id]
    
    def list_active_calls(self) -> List[str]:
        """
        Get list of all active call IDs.
        
        Returns:
            List of call IDs with active memory
        """
        with self._lock:
            return list(self._memories.keys())
    
    def clear_all(self) -> int:
        """
        Clear all call memories (use with caution).
        
        Returns:
            Number of memories cleared
        """
        with self._lock:
            count = len(self._memories)
            self._memories.clear()
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory store.
        
        Returns:
            Dictionary with store statistics
        """
        with self._lock:
            return {
                "active_calls": len(self._memories),
                "call_ids": list(self._memories.keys()),
                "total_turns": sum(
                    mem.turn_count for mem in self._memories.values()
                )
            }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

# Global memory store instance
_store = MemoryStore()


# ============================================================================
# PUBLIC API
# ============================================================================

def create_call_memory(call_id: str) -> CallMemory:
    """
    Create a new call memory instance.
    
    Args:
        call_id: Unique identifier for the call
        
    Returns:
        New CallMemory instance
        
    Example:
        >>> memory = create_call_memory("CA123abc")
        >>> memory.set_restaurant_id("rest_123")
    """
    return _store.create_call_memory(call_id)


def get_memory(call_id: str) -> Optional[CallMemory]:
    """
    Retrieve memory for a specific call.
    
    Args:
        call_id: Call identifier
        
    Returns:
        CallMemory instance or None if not found
        
    Example:
        >>> memory = get_memory("CA123abc")
        >>> if memory:
        >>>     memory.add_order_item("pizza_1", "Margherita Pizza", 1, 12.99)
    """
    return _store.get_memory(call_id)


def clear_memory(call_id: str) -> bool:
    """
    Clear and remove memory for a call.
    Should be called when call ends.
    
    Args:
        call_id: Call identifier
        
    Returns:
        True if memory was cleared, False if not found
        
    Example:
        >>> clear_memory("CA123abc")
        True
    """
    return _store.clear_memory(call_id)


def get_or_create_memory(call_id: str) -> CallMemory:
    """
    Get existing memory or create new one if doesn't exist.
    Convenience function for simple workflows.
    
    Args:
        call_id: Call identifier
        
    Returns:
        CallMemory instance
        
    Example:
        >>> memory = get_or_create_memory("CA123abc")
    """
    return _store.get_or_create(call_id)


def list_active_calls() -> List[str]:
    """
    Get list of all active call IDs.
    
    Returns:
        List of call IDs with active memory
        
    Example:
        >>> active = list_active_calls()
        >>> print(f"Active calls: {len(active)}")
    """
    return _store.list_active_calls()


def get_stats() -> Dict[str, Any]:
    """
    Get statistics about the memory store.
    
    Returns:
        Dictionary with store statistics
        
    Example:
        >>> stats = get_stats()
        >>> print(f"Active calls: {stats['active_calls']}")
    """
    return _store.get_stats()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the memory engine.
    """
    
    # Create memory for new call
    call_id = "CA123abc456def"
    memory = create_call_memory(call_id)
    
    # Set restaurant
    memory.set_restaurant_id("rest_123")
    
    # Set customer info
    memory.set_customer_info(phone="+1234567890", name="John Doe")
    
    # Detect language
    memory.update_language("en")
    
    # Store menu snapshot
    memory.set_menu_snapshot([
        {"id": "pizza_1", "name": "Margherita", "price": 12.99},
        {"id": "pizza_2", "name": "Pepperoni", "price": 14.99}
    ])
    
    # Track conversation
    memory.add_conversation_turn(
        role="user",
        content="I'd like to order a pizza",
        intent="start_order"
    )
    memory.set_last_intent("start_order")
    
    memory.add_conversation_turn(
        role="assistant",
        content="Great! What type of pizza would you like?",
        intent=None
    )
    
    memory.add_conversation_turn(
        role="user",
        content="I'll have a Margherita pizza",
        intent="add_item"
    )
    memory.set_last_intent("add_item")
    
    # Add items to order
    memory.add_order_item(
        item_id="pizza_1",
        name="Margherita Pizza",
        quantity=1,
        price=12.99,
        customizations=["extra cheese"]
    )
    
    memory.add_order_item(
        item_id="drink_1",
        name="Coke",
        quantity=2,
        price=2.50
    )
    
    # Get order summary
    summary = memory.get_order_summary()
    print(f"\nOrder Summary:")
    print(f"Items: {summary['item_count']}")
    print(f"Subtotal: ${summary['subtotal']:.2f}")
    print(f"Tax: ${summary['tax']:.2f}")
    print(f"Total: ${summary['total']:.2f}")
    
    # Get recent context
    recent = memory.get_recent_context(num_turns=3)
    print(f"\nRecent conversation ({len(recent)} turns):")
    for turn in recent:
        print(f"  {turn['role']}: {turn['content']}")
    
    # Update order
    memory.update_order_item_quantity("drink_1", 3)
    
    # Get updated summary
    summary = memory.get_order_summary()
    print(f"\nUpdated Total: ${summary['total']:.2f}")
    
    # Convert to dict for logging/debugging
    data = memory.to_dict()
    print(f"\nMemory state: {data['state']}")
    print(f"Turn count: {data['turn_count']}")
    
    # Retrieve memory later
    retrieved = get_memory(call_id)
    if retrieved:
        print(f"\nRetrieved memory for call: {retrieved.call_id}")
    
    # Check stats
    stats = get_stats()
    print(f"\nActive calls: {stats['active_calls']}")
    
    # Clean up when call ends
    cleared = clear_memory(call_id)
    print(f"\nMemory cleared: {cleared}")
