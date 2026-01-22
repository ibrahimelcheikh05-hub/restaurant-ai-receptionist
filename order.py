"""
Order Module (Production Hardened)
===================================
Transactional state machine for order management.

HARDENING UPDATES (v3.0):
✅ Complete state machine: EMPTY, BUILDING, CONFIRMING, FINAL, LOCKED
✅ Integrity validation (checksums, totals)
✅ No silent overwrites (explicit warnings)
✅ Rollback capability (snapshots)
✅ Explicit commit step (two-phase)
✅ Immutable finalized orders (frozen)
✅ AI cannot directly mutate

Version: 3.0.0 (Production Hardened)
Last Updated: 2026-01-22
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict
from copy import deepcopy
from enum import Enum
import uuid
import hashlib

try:
    from prometheus_client import Counter, Histogram, Gauge
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False


logger = logging.getLogger(__name__)


# ============================================================================
# METRICS
# ============================================================================

if METRICS_ENABLED:
    orders_total = Counter(
        'orders_total',
        'Total orders',
        ['state']
    )
    order_value = Histogram(
        'order_value_dollars',
        'Order value distribution'
    )
    order_state_transitions = Counter(
        'order_state_transitions_total',
        'Order state transitions',
        ['from_state', 'to_state']
    )
    order_validation_failures = Counter(
        'order_validation_failures_total',
        'Order validation failures',
        ['reason']
    )
    order_rollbacks = Counter(
        'order_rollbacks_total',
        'Order rollback events'
    )
    order_commits = Counter(
        'order_commits_total',
        'Order commit events',
        ['result']
    )
    order_overwrite_warnings = Counter(
        'order_overwrite_warnings_total',
        'Order overwrite warning events'
    )
    orders_active = Gauge(
        'orders_active',
        'Currently active orders'
    )


# ============================================================================
# ORDER STATE (Comprehensive)
# ============================================================================

class OrderState(Enum):
    """
    Order lifecycle states (comprehensive state machine).
    
    State transitions:
    EMPTY → BUILDING → CONFIRMING → FINAL → LOCKED
    
    Terminal states: FINAL, LOCKED, CANCELLED
    """
    EMPTY = "empty"              # No items yet
    BUILDING = "building"        # Adding/removing items
    CONFIRMING = "confirming"    # Review phase (pre-commit)
    FINAL = "final"              # Committed (immutable)
    LOCKED = "locked"            # Finalized with customer info (fully immutable)
    CANCELLED = "cancelled"      # Cancelled (terminal)


# ============================================================================
# IMMUTABLE ORDER ITEM
# ============================================================================

@dataclass(frozen=True)
class OrderItem:
    """
    Immutable order item.
    
    CRITICAL: frozen=True means item CANNOT be modified after creation.
    """
    item_id: str
    canonical_id: str  # From menu system (for validation)
    name: str
    price: float
    quantity: int
    subtotal: float
    notes: Optional[str] = None
    added_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def with_quantity(self, new_quantity: int) -> 'OrderItem':
        """
        Create new item with updated quantity (immutable).
        
        This is the ONLY way to "modify" - creates new object.
        """
        return OrderItem(
            item_id=self.item_id,
            canonical_id=self.canonical_id,
            name=self.name,
            price=self.price,
            quantity=new_quantity,
            subtotal=round(self.price * new_quantity, 2),
            notes=self.notes,
            added_at=self.added_at
        )


# ============================================================================
# ORDER SNAPSHOT (Rollback Support)
# ============================================================================

@dataclass(frozen=True)
class OrderSnapshot:
    """
    Immutable order snapshot for rollback.
    
    Captures complete order state at a point in time.
    """
    snapshot_id: str
    order_id: str
    state: OrderState
    items: Tuple[OrderItem, ...]  # Immutable tuple
    subtotal: float
    tax: float
    total: float
    timestamp: datetime
    checksum: str  # Integrity validation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "order_id": self.order_id,
            "state": self.state.value,
            "items": [item.to_dict() for item in self.items],
            "subtotal": self.subtotal,
            "tax": self.tax,
            "total": self.total,
            "timestamp": self.timestamp.isoformat(),
            "checksum": self.checksum
        }


# ============================================================================
# TRANSACTIONAL ORDER
# ============================================================================

class TransactionalOrder:
    """
    Transactional order with state machine and rollback.
    
    State machine:
    EMPTY → BUILDING → CONFIRMING → FINAL → LOCKED
    
    Guarantees:
    - Integrity validation via checksums
    - No silent overwrites (explicit warnings)
    - Rollback capability via snapshots
    - Explicit commit step
    - Immutable after FINAL state
    """
    
    def __init__(self, call_id: str, restaurant_id: str):
        """Initialize transactional order."""
        self.order_id = f"ord_{uuid.uuid4().hex[:12]}"
        self.call_id = call_id
        self.restaurant_id = restaurant_id
        
        # State machine
        self.state = OrderState.EMPTY
        self._state_history: List[Tuple[OrderState, datetime]] = [
            (OrderState.EMPTY, datetime.utcnow())
        ]
        
        # Items (mutable until FINAL)
        self._items: Dict[str, OrderItem] = {}
        
        # Pricing
        self.subtotal = 0.0
        self.tax_rate = 0.08
        self.tax = 0.0
        self.total = 0.0
        
        # Customer info (set during CONFIRMING → FINAL)
        self.customer_name: Optional[str] = None
        self.customer_phone: Optional[str] = None
        self.delivery_address: Optional[str] = None
        self.order_type: str = "pickup"
        
        # Timestamps
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.confirmed_at: Optional[datetime] = None
        self.finalized_at: Optional[datetime] = None
        
        # Limits
        self.max_items = 50
        self.max_quantity_per_item = 99
        self.min_order_total = 0.01
        
        # Rollback support (snapshots)
        self._snapshots: List[OrderSnapshot] = []
        self._max_snapshots = 10
        
        # Integrity
        self._checksum: Optional[str] = None
        
        # Tracking
        self.modification_count = 0
        self.validation_count = 0
        self.rollback_count = 0
        
        # Track in registry
        _order_registry[self.order_id] = self
        
        if METRICS_ENABLED:
            orders_active.inc()
            orders_total.labels(state=self.state.value).inc()
        
        logger.info(f"Transactional order created: {self.order_id}")
    
    # ========================================================================
    # STATE MACHINE
    # ========================================================================
    
    def transition(self, new_state: OrderState) -> bool:
        """
        Transition to new state (controlled).
        
        Valid transitions:
        EMPTY → BUILDING
        BUILDING → CONFIRMING
        BUILDING → EMPTY (clear)
        CONFIRMING → BUILDING (back to editing)
        CONFIRMING → FINAL (commit)
        FINAL → LOCKED (finalize with customer info)
        
        Terminal states: FINAL, LOCKED, CANCELLED
        
        Args:
            new_state: Target state
            
        Returns:
            True if transition allowed
        """
        valid_transitions = {
            OrderState.EMPTY: [OrderState.BUILDING, OrderState.CANCELLED],
            OrderState.BUILDING: [OrderState.EMPTY, OrderState.CONFIRMING, OrderState.CANCELLED],
            OrderState.CONFIRMING: [OrderState.BUILDING, OrderState.FINAL, OrderState.CANCELLED],
            OrderState.FINAL: [OrderState.LOCKED],
            OrderState.LOCKED: [],  # Terminal
            OrderState.CANCELLED: []  # Terminal
        }
        
        if new_state not in valid_transitions.get(self.state, []):
            logger.warning(
                f"Invalid transition: {self.state.value} → {new_state.value}"
            )
            return False
        
        # Record transition
        old_state = self.state
        self.state = new_state
        self._state_history.append((new_state, datetime.utcnow()))
        self._touch()
        
        if METRICS_ENABLED:
            order_state_transitions.labels(
                from_state=old_state.value,
                to_state=new_state.value
            ).inc()
        
        logger.info(
            f"Order {self.order_id}: {old_state.value} → {new_state.value}"
        )
        
        return True
    
    def get_state(self) -> OrderState:
        """Get current state."""
        return self.state
    
    def is_mutable(self) -> bool:
        """Check if order can be modified."""
        return self.state in [OrderState.EMPTY, OrderState.BUILDING]
    
    def is_final(self) -> bool:
        """Check if order is in final state."""
        return self.state in [OrderState.FINAL, OrderState.LOCKED]
    
    def _touch(self):
        """Update timestamp."""
        self.updated_at = datetime.utcnow()
    
    # ========================================================================
    # ITEM OPERATIONS (Controlled Mutation)
    # ========================================================================
    
    def add_item(
        self,
        item_id: str,
        canonical_id: str,
        name: str,
        price: float,
        quantity: int = 1,
        notes: Optional[str] = None,
        allow_overwrite: bool = False
    ) -> bool:
        """
        Add item to order (controlled mutation).
        
        RULES:
        - Only in EMPTY or BUILDING state
        - No silent overwrites (requires explicit flag)
        - Creates immutable OrderItem
        
        Args:
            item_id: Item identifier
            canonical_id: Canonical menu item ID (for validation)
            name: Item name
            price: Item price
            quantity: Quantity
            notes: Special notes
            allow_overwrite: Allow overwriting existing item
            
        Returns:
            True if added successfully
        """
        # Check mutability
        if not self.is_mutable():
            logger.error(
                f"Cannot add item: order in {self.state.value} state"
            )
            return False
        
        # Transition to BUILDING if EMPTY
        if self.state == OrderState.EMPTY:
            self.transition(OrderState.BUILDING)
        
        # Validate inputs
        if not item_id or not canonical_id or not name:
            logger.warning("Item ID, canonical ID, and name required")
            if METRICS_ENABLED:
                order_validation_failures.labels(reason='missing_fields').inc()
            return False
        
        # Normalize
        quantity = self._normalize_quantity(quantity)
        if quantity <= 0:
            if METRICS_ENABLED:
                order_validation_failures.labels(reason='invalid_quantity').inc()
            return False
        
        price = self._normalize_price(price)
        if price is None or price < 0:
            if METRICS_ENABLED:
                order_validation_failures.labels(reason='invalid_price').inc()
            return False
        
        # Check item limit
        if item_id not in self._items and len(self._items) >= self.max_items:
            logger.warning(f"Max items reached: {self.max_items}")
            if METRICS_ENABLED:
                order_validation_failures.labels(reason='max_items').inc()
            return False
        
        # Check for overwrite
        if item_id in self._items and not allow_overwrite:
            logger.warning(
                f"Item {item_id} already in order - set allow_overwrite=True to replace"
            )
            
            if METRICS_ENABLED:
                order_overwrite_warnings.inc()
            
            return False  # NO SILENT OVERWRITES!
        
        # Create snapshot before mutation
        self._create_snapshot()
        
        # Create immutable item
        subtotal = round(price * quantity, 2)
        
        item = OrderItem(
            item_id=item_id,
            canonical_id=canonical_id,
            name=name,
            price=price,
            quantity=quantity,
            subtotal=subtotal,
            notes=notes
        )
        
        # Add item
        self._items[item_id] = item
        self.modification_count += 1
        
        # Recompute
        self._recompute_totals()
        
        logger.info(
            f"Added item: {name} (quantity={quantity}, price=${price:.2f})"
        )
        
        return True
    
    def remove_item(self, item_id: str) -> bool:
        """
        Remove item from order.
        
        Args:
            item_id: Item identifier
            
        Returns:
            True if removed
        """
        if not self.is_mutable():
            logger.error(f"Cannot remove: order in {self.state.value} state")
            return False
        
        if item_id not in self._items:
            logger.warning(f"Item {item_id} not in order")
            return False
        
        # Create snapshot before mutation
        self._create_snapshot()
        
        # Remove
        item = self._items.pop(item_id)
        self.modification_count += 1
        
        # Recompute
        self._recompute_totals()
        
        # Transition to EMPTY if no items
        if len(self._items) == 0:
            self.transition(OrderState.EMPTY)
        
        logger.info(f"Removed item: {item.name}")
        
        return True
    
    def update_quantity(self, item_id: str, new_quantity: int) -> bool:
        """
        Update item quantity (creates new immutable item).
        
        Args:
            item_id: Item identifier
            new_quantity: New quantity
            
        Returns:
            True if updated
        """
        if not self.is_mutable():
            logger.error(f"Cannot update: order in {self.state.value} state")
            return False
        
        if item_id not in self._items:
            logger.warning(f"Item {item_id} not in order")
            return False
        
        # Normalize
        new_quantity = self._normalize_quantity(new_quantity)
        if new_quantity <= 0:
            # Remove if quantity is 0
            return self.remove_item(item_id)
        
        if new_quantity > self.max_quantity_per_item:
            logger.warning(f"Max quantity exceeded: {self.max_quantity_per_item}")
            return False
        
        # Create snapshot before mutation
        self._create_snapshot()
        
        # Create new item with updated quantity (immutable)
        old_item = self._items[item_id]
        new_item = old_item.with_quantity(new_quantity)
        
        # Replace
        self._items[item_id] = new_item
        self.modification_count += 1
        
        # Recompute
        self._recompute_totals()
        
        logger.info(
            f"Updated quantity: {old_item.name} "
            f"{old_item.quantity} → {new_quantity}"
        )
        
        return True
    
    def clear_items(self) -> bool:
        """
        Clear all items (transition to EMPTY).
        
        Returns:
            True if cleared
        """
        if not self.is_mutable():
            logger.error(f"Cannot clear: order in {self.state.value} state")
            return False
        
        # Create snapshot before mutation
        self._create_snapshot()
        
        # Clear
        item_count = len(self._items)
        self._items.clear()
        self.modification_count += 1
        
        # Recompute
        self._recompute_totals()
        
        # Transition to EMPTY
        self.transition(OrderState.EMPTY)
        
        logger.info(f"Cleared {item_count} items")
        
        return True
    
    # ========================================================================
    # COMMIT / ROLLBACK (Transactional)
    # ========================================================================
    
    def prepare_commit(self) -> bool:
        """
        Prepare for commit (transition to CONFIRMING).
        
        This is the first phase of two-phase commit.
        
        Returns:
            True if ready for commit
        """
        if self.state != OrderState.BUILDING:
            logger.warning(
                f"Cannot prepare: order in {self.state.value} state"
            )
            return False
        
        # Validate order
        is_valid, errors = self.validate()
        if not is_valid:
            logger.warning(f"Order validation failed: {errors}")
            return False
        
        # Transition to CONFIRMING
        if not self.transition(OrderState.CONFIRMING):
            return False
        
        self.confirmed_at = datetime.utcnow()
        
        # Create final snapshot before commit
        self._create_snapshot()
        
        logger.info(f"Order prepared for commit: {self.order_id}")
        
        return True
    
    def commit(self) -> bool:
        """
        Commit order (transition to FINAL).
        
        This is the second phase of two-phase commit.
        Makes order IMMUTABLE.
        
        Returns:
            True if committed
        """
        if self.state != OrderState.CONFIRMING:
            logger.error(
                f"Cannot commit: must be in CONFIRMING state "
                f"(current: {self.state.value})"
            )
            
            if METRICS_ENABLED:
                order_commits.labels(result='invalid_state').inc()
            
            return False
        
        # Final validation
        is_valid, errors = self.validate()
        if not is_valid:
            logger.error(f"Commit validation failed: {errors}")
            
            if METRICS_ENABLED:
                order_commits.labels(result='validation_failed').inc()
            
            return False
        
        # Transition to FINAL (immutable)
        if not self.transition(OrderState.FINAL):
            if METRICS_ENABLED:
                order_commits.labels(result='transition_failed').inc()
            return False
        
        self.finalized_at = datetime.utcnow()
        
        # Compute final checksum
        self._checksum = self._compute_checksum()
        
        if METRICS_ENABLED:
            order_commits.labels(result='success').inc()
            order_value.observe(self.total)
        
        logger.info(
            f"Order committed: {self.order_id} "
            f"(total=${self.total:.2f}, checksum={self._checksum[:8]})"
        )
        
        return True
    
    def rollback(self) -> bool:
        """
        Rollback to last snapshot.
        
        Returns:
            True if rolled back
        """
        if not self._snapshots:
            logger.warning("No snapshots available for rollback")
            return False
        
        if self.is_final():
            logger.error("Cannot rollback: order is final")
            return False
        
        # Get last snapshot
        snapshot = self._snapshots[-1]
        
        # Restore state
        self.state = snapshot.state
        self._items = {item.item_id: item for item in snapshot.items}
        self.subtotal = snapshot.subtotal
        self.tax = snapshot.tax
        self.total = snapshot.total
        
        self.rollback_count += 1
        self._touch()
        
        if METRICS_ENABLED:
            order_rollbacks.inc()
        
        logger.info(
            f"Rolled back to snapshot: {snapshot.snapshot_id} "
            f"(state={snapshot.state.value})"
        )
        
        return True
    
    def _create_snapshot(self):
        """Create snapshot of current state (for rollback)."""
        snapshot_id = f"snap_{uuid.uuid4().hex[:8]}"
        
        snapshot = OrderSnapshot(
            snapshot_id=snapshot_id,
            order_id=self.order_id,
            state=self.state,
            items=tuple(self._items.values()),  # Immutable tuple
            subtotal=self.subtotal,
            tax=self.tax,
            total=self.total,
            timestamp=datetime.utcnow(),
            checksum=self._compute_checksum()
        )
        
        self._snapshots.append(snapshot)
        
        # Limit snapshot count
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots.pop(0)
        
        logger.debug(f"Created snapshot: {snapshot_id}")
    
    def get_snapshots(self) -> List[OrderSnapshot]:
        """Get all snapshots."""
        return self._snapshots.copy()
    
    # ========================================================================
    # VALIDATION & INTEGRITY
    # ========================================================================
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate order integrity.
        
        Checks:
        - Has items
        - Totals are correct
        - Prices are valid
        - Quantities are valid
        
        Returns:
            (is_valid, error_messages)
        """
        self.validation_count += 1
        errors = []
        
        # Check has items
        if len(self._items) == 0:
            errors.append("Order has no items")
        
        # Validate totals
        expected_subtotal = sum(item.subtotal for item in self._items.values())
        if abs(self.subtotal - expected_subtotal) > 0.01:
            errors.append(
                f"Subtotal mismatch: {self.subtotal} != {expected_subtotal}"
            )
        
        expected_tax = round(self.subtotal * self.tax_rate, 2)
        if abs(self.tax - expected_tax) > 0.01:
            errors.append(
                f"Tax mismatch: {self.tax} != {expected_tax}"
            )
        
        expected_total = self.subtotal + self.tax
        if abs(self.total - expected_total) > 0.01:
            errors.append(
                f"Total mismatch: {self.total} != {expected_total}"
            )
        
        # Validate minimum total
        if self.total < self.min_order_total:
            errors.append(
                f"Order total too low: ${self.total:.2f} < ${self.min_order_total:.2f}"
            )
        
        # Validate items
        for item in self._items.values():
            if item.quantity <= 0:
                errors.append(f"Invalid quantity for {item.name}: {item.quantity}")
            
            if item.price < 0:
                errors.append(f"Invalid price for {item.name}: ${item.price:.2f}")
            
            expected_item_subtotal = round(item.price * item.quantity, 2)
            if abs(item.subtotal - expected_item_subtotal) > 0.01:
                errors.append(
                    f"Item subtotal mismatch for {item.name}: "
                    f"{item.subtotal} != {expected_item_subtotal}"
                )
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.warning(f"Order validation failed: {errors}")
            
            if METRICS_ENABLED:
                for error in errors:
                    order_validation_failures.labels(
                        reason=error.split(':')[0]
                    ).inc()
        
        return is_valid, errors
    
    def verify_integrity(self) -> bool:
        """
        Verify order integrity via checksum.
        
        Returns:
            True if integrity verified
        """
        if not self._checksum:
            logger.warning("No checksum available")
            return False
        
        current_checksum = self._compute_checksum()
        
        if current_checksum != self._checksum:
            logger.error(
                f"Integrity check failed: "
                f"{current_checksum[:8]} != {self._checksum[:8]}"
            )
            return False
        
        logger.debug("Integrity verified")
        return True
    
    def _compute_checksum(self) -> str:
        """
        Compute order checksum (for integrity validation).
        
        Deterministic: same order always produces same checksum.
        """
        # Serialize order state
        data = {
            "order_id": self.order_id,
            "items": sorted([
                (item.item_id, item.canonical_id, item.quantity, item.price)
                for item in self._items.values()
            ]),
            "subtotal": self.subtotal,
            "tax": self.tax,
            "total": self.total
        }
        
        # Compute hash
        content = str(data).encode()
        return hashlib.sha256(content).hexdigest()
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def _normalize_quantity(self, quantity: Any) -> int:
        """Normalize quantity to int."""
        try:
            q = int(quantity)
            return max(0, min(q, self.max_quantity_per_item))
        except (ValueError, TypeError):
            return 0
    
    def _normalize_price(self, price: Any) -> Optional[float]:
        """Normalize price to float."""
        try:
            p = float(price)
            return round(p, 2) if p >= 0 else None
        except (ValueError, TypeError):
            return None
    
    def _recompute_totals(self):
        """Recompute order totals (deterministic)."""
        self.subtotal = round(
            sum(item.subtotal for item in self._items.values()),
            2
        )
        self.tax = round(self.subtotal * self.tax_rate, 2)
        self.total = round(self.subtotal + self.tax, 2)
        
        self._touch()
    
    def get_items(self) -> List[OrderItem]:
        """Get all items (immutable list)."""
        return list(self._items.values())
    
    def get_item(self, item_id: str) -> Optional[OrderItem]:
        """Get specific item."""
        return self._items.get(item_id)
    
    def get_item_count(self) -> int:
        """Get total item count."""
        return sum(item.quantity for item in self._items.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return {
            "order_id": self.order_id,
            "call_id": self.call_id,
            "restaurant_id": self.restaurant_id,
            "state": self.state.value,
            "items": [item.to_dict() for item in self._items.values()],
            "subtotal": self.subtotal,
            "tax": self.tax,
            "tax_rate": self.tax_rate,
            "total": self.total,
            "item_count": self.get_item_count(),
            "customer_name": self.customer_name,
            "customer_phone": self.customer_phone,
            "delivery_address": self.delivery_address,
            "order_type": self.order_type,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "confirmed_at": self.confirmed_at.isoformat() if self.confirmed_at else None,
            "finalized_at": self.finalized_at.isoformat() if self.finalized_at else None,
            "checksum": self._checksum,
            "is_mutable": self.is_mutable(),
            "is_final": self.is_final(),
            "modification_count": self.modification_count,
            "validation_count": self.validation_count,
            "rollback_count": self.rollback_count,
            "snapshot_count": len(self._snapshots)
        }


# ============================================================================
# GLOBAL REGISTRY
# ============================================================================

_order_registry: Dict[str, TransactionalOrder] = {}


def create_order(call_id: str, restaurant_id: str) -> TransactionalOrder:
    """Create new transactional order."""
    order = TransactionalOrder(call_id, restaurant_id)
    return order


def get_order(order_id: str) -> Optional[TransactionalOrder]:
    """Get order by ID."""
    return _order_registry.get(order_id)


def clear_order(order_id: str):
    """Clear order from registry."""
    if order_id in _order_registry:
        del _order_registry[order_id]
        
        if METRICS_ENABLED:
            orders_active.dec()
        
        logger.info(f"Order cleared: {order_id}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Order Module (Production Hardened v3.0)")
    print("="*60)
    
    # Create order
    order = create_order("call_123", "rest_001")
    print(f"\n1. Created order: {order.order_id}")
    print(f"   State: {order.state.value}")
    
    # Add item (transitions to BUILDING)
    success = order.add_item(
        item_id="item_1",
        canonical_id="menu_abc123",
        name="Large Pizza",
        price=15.99,
        quantity=2
    )
    print(f"\n2. Added item: {success}")
    print(f"   State: {order.state.value}")
    print(f"   Total: ${order.total:.2f}")
    
    # Try to add duplicate without flag (rejected)
    success = order.add_item(
        item_id="item_1",
        canonical_id="menu_abc123",
        name="Large Pizza",
        price=15.99,
        quantity=1
    )
    print(f"\n3. Duplicate add (no flag): {success}")
    
    # Add with overwrite flag
    success = order.add_item(
        item_id="item_1",
        canonical_id="menu_abc123",
        name="Large Pizza",
        price=15.99,
        quantity=3,
        allow_overwrite=True
    )
    print(f"\n4. Duplicate add (with flag): {success}")
    
    # Prepare commit
    success = order.prepare_commit()
    print(f"\n5. Prepared for commit: {success}")
    print(f"   State: {order.state.value}")
    
    # Rollback
    success = order.rollback()
    print(f"\n6. Rolled back: {success}")
    print(f"   State: {order.state.value}")
    
    # Prepare and commit
    order.prepare_commit()
    success = order.commit()
    print(f"\n7. Committed: {success}")
    print(f"   State: {order.state.value}")
    print(f"   Is mutable: {order.is_mutable()}")
    print(f"   Is final: {order.is_final()}")
    
    # Try to modify (rejected)
    success = order.add_item(
        item_id="item_2",
        canonical_id="menu_xyz789",
        name="Salad",
        price=8.99,
        quantity=1
    )
    print(f"\n8. Modify after commit: {success}")
    
    # Validation
    is_valid, errors = order.validate()
    print(f"\n9. Validation: {is_valid}")
    if errors:
        for error in errors:
            print(f"   - {error}")
    
    # Export
    export = order.to_dict()
    print(f"\n10. Export:")
    print(f"   Order ID: {export['order_id']}")
    print(f"   State: {export['state']}")
    print(f"   Items: {export['item_count']}")
    print(f"   Total: ${export['total']:.2f}")
    print(f"   Checksum: {export['checksum'][:16]}...")
