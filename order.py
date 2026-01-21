"""
Order Module (Production)
==========================
Hardened transactional order management with state machine.
Deterministic state, validation, quantity normalization, finalization locks.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid


logger = logging.getLogger(__name__)


class OrderState(Enum):
    """Order lifecycle states."""
    CREATED = "created"
    BUILDING = "building"
    VALIDATING = "validating"
    FINALIZED = "finalized"
    CANCELLED = "cancelled"


@dataclass
class OrderItem:
    """Single order item."""
    item_id: str
    name: str
    price: float
    quantity: int
    subtotal: float
    notes: Optional[str] = None
    added_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class OrderSummary:
    """Complete order summary."""
    order_id: str
    call_id: str
    restaurant_id: str
    state: str
    items: List[OrderItem]
    subtotal: float
    tax: float
    total: float
    item_count: int
    created_at: str
    updated_at: str
    finalized_at: Optional[str] = None


class Order:
    """
    Transactional order with deterministic state machine.
    Supports safe add/remove, validation, and finalization.
    """
    
    def __init__(self, call_id: str, restaurant_id: str):
        """
        Initialize order.
        
        Args:
            call_id: Call identifier
            restaurant_id: Restaurant identifier
        """
        self.order_id = f"ord_{uuid.uuid4().hex[:12]}"
        self.call_id = call_id
        self.restaurant_id = restaurant_id
        self.state = OrderState.CREATED
        
        # Items
        self.items: Dict[str, OrderItem] = {}
        
        # Pricing
        self.subtotal = 0.0
        self.tax_rate = 0.08  # 8% default
        self.tax = 0.0
        self.total = 0.0
        
        # Customer info (set on finalization)
        self.customer_name: Optional[str] = None
        self.customer_phone: Optional[str] = None
        self.delivery_address: Optional[str] = None
        self.order_type: str = "pickup"  # "pickup" or "delivery"
        
        # Timestamps
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.finalized_at: Optional[datetime] = None
        
        # Limits
        self.max_items = 50
        self.max_quantity_per_item = 99
        self.min_order_total = 0.01
        
        # Lock flags
        self._finalized = False
        
        logger.info(f"Order created: {self.order_id} for call {call_id}")
    
    # ========================================================================
    # STATE MANAGEMENT
    # ========================================================================
    
    def _transition(self, new_state: OrderState) -> bool:
        """
        Transition to new state with validation.
        
        Args:
            new_state: Target state
            
        Returns:
            True if transition allowed
        """
        valid_transitions = {
            OrderState.CREATED: [OrderState.BUILDING, OrderState.CANCELLED],
            OrderState.BUILDING: [OrderState.VALIDATING, OrderState.CANCELLED],
            OrderState.VALIDATING: [OrderState.BUILDING, OrderState.FINALIZED, OrderState.CANCELLED],
            OrderState.FINALIZED: [],  # Terminal state
            OrderState.CANCELLED: []   # Terminal state
        }
        
        if new_state in valid_transitions.get(self.state, []):
            old_state = self.state
            self.state = new_state
            self._touch()
            logger.debug(f"Order {self.order_id}: {old_state.value} → {new_state.value}")
            return True
        
        logger.warning(
            f"Invalid transition for order {self.order_id}: "
            f"{self.state.value} → {new_state.value}"
        )
        return False
    
    def _touch(self):
        """Update timestamp."""
        self.updated_at = datetime.utcnow()
    
    def _check_finalized(self) -> bool:
        """Check if order is finalized (immutable)."""
        if self._finalized or self.state == OrderState.FINALIZED:
            logger.warning(f"Order {self.order_id} is finalized, cannot modify")
            return True
        return False
    
    # ========================================================================
    # ITEM MANAGEMENT
    # ========================================================================
    
    def add_item(
        self,
        item_id: str,
        name: str,
        price: float,
        quantity: int = 1,
        notes: Optional[str] = None
    ) -> bool:
        """
        Add item to order with validation.
        
        Args:
            item_id: Item identifier
            name: Item name
            price: Item price
            quantity: Quantity (default: 1)
            notes: Special notes (optional)
            
        Returns:
            True if added successfully
        """
        if self._check_finalized():
            return False
        
        # Transition to building state
        if self.state == OrderState.CREATED:
            self._transition(OrderState.BUILDING)
        
        # Validate inputs
        if not item_id or not name:
            logger.warning("Item ID and name required")
            return False
        
        # Normalize quantity
        quantity = self._normalize_quantity(quantity)
        if quantity <= 0:
            logger.warning(f"Invalid quantity: {quantity}")
            return False
        
        # Validate price
        price = self._normalize_price(price)
        if price is None or price < 0:
            logger.warning(f"Invalid price: {price}")
            return False
        
        # Check item limit
        if item_id not in self.items and len(self.items) >= self.max_items:
            logger.warning(f"Max items reached ({self.max_items})")
            return False
        
        # Calculate subtotal
        subtotal = round(price * quantity, 2)
        
        # Add or update item
        if item_id in self.items:
            # Update existing item (add to quantity)
            existing = self.items[item_id]
            new_quantity = existing.quantity + quantity
            
            if new_quantity > self.max_quantity_per_item:
                logger.warning(f"Max quantity exceeded for item {name}")
                return False
            
            existing.quantity = new_quantity
            existing.subtotal = round(price * new_quantity, 2)
            existing.notes = notes if notes else existing.notes
            
            logger.info(f"Updated item {name}: quantity={new_quantity}")
        else:
            # Add new item
            self.items[item_id] = OrderItem(
                item_id=item_id,
                name=name,
                price=price,
                quantity=quantity,
                subtotal=subtotal,
                notes=notes
            )
            
            logger.info(f"Added item {name}: {quantity}x ${price:.2f}")
        
        # Recompute totals
        self._recompute_totals()
        
        return True
    
    def remove_item(self, item_id: str) -> bool:
        """
        Remove item from order.
        
        Args:
            item_id: Item identifier
            
        Returns:
            True if removed successfully
        """
        if self._check_finalized():
            return False
        
        if item_id not in self.items:
            logger.warning(f"Item {item_id} not in order")
            return False
        
        item = self.items.pop(item_id)
        logger.info(f"Removed item {item.name} from order {self.order_id}")
        
        # Recompute totals
        self._recompute_totals()
        
        return True
    
    def update_item_quantity(self, item_id: str, quantity: int) -> bool:
        """
        Update item quantity.
        
        Args:
            item_id: Item identifier
            quantity: New quantity
            
        Returns:
            True if updated successfully
        """
        if self._check_finalized():
            return False
        
        if item_id not in self.items:
            logger.warning(f"Item {item_id} not in order")
            return False
        
        # Normalize quantity
        quantity = self._normalize_quantity(quantity)
        
        if quantity <= 0:
            # Remove item if quantity is 0
            return self.remove_item(item_id)
        
        if quantity > self.max_quantity_per_item:
            logger.warning(f"Quantity exceeds maximum: {quantity}")
            return False
        
        item = self.items[item_id]
        item.quantity = quantity
        item.subtotal = round(item.price * quantity, 2)
        
        logger.info(f"Updated quantity for {item.name}: {quantity}")
        
        # Recompute totals
        self._recompute_totals()
        
        return True
    
    def clear_items(self):
        """Clear all items from order."""
        if self._check_finalized():
            return
        
        count = len(self.items)
        self.items.clear()
        
        logger.info(f"Cleared {count} items from order {self.order_id}")
        
        # Recompute totals
        self._recompute_totals()
    
    # ========================================================================
    # VALIDATION & TOTALS
    # ========================================================================
    
    def _normalize_quantity(self, quantity: Any) -> int:
        """
        Normalize quantity to integer.
        
        Args:
            quantity: Raw quantity value
            
        Returns:
            Normalized quantity
        """
        try:
            qty = int(quantity)
            return max(0, min(qty, self.max_quantity_per_item))
        except (ValueError, TypeError):
            logger.warning(f"Invalid quantity: {quantity}")
            return 0
    
    def _normalize_price(self, price: Any) -> Optional[float]:
        """
        Normalize price to float.
        
        Args:
            price: Raw price value
            
        Returns:
            Normalized price or None if invalid
        """
        try:
            if isinstance(price, (int, float)):
                price_float = float(price)
            elif isinstance(price, str):
                price_clean = price.replace("$", "").replace(",", "").strip()
                price_float = float(price_clean)
            else:
                return None
            
            return round(max(0.0, price_float), 2)
        
        except (ValueError, TypeError):
            logger.warning(f"Invalid price: {price}")
            return None
    
    def _recompute_totals(self):
        """Recompute order totals."""
        # Calculate subtotal
        self.subtotal = sum(item.subtotal for item in self.items.values())
        self.subtotal = round(self.subtotal, 2)
        
        # Calculate tax
        self.tax = round(self.subtotal * self.tax_rate, 2)
        
        # Calculate total
        self.total = round(self.subtotal + self.tax, 2)
        
        self._touch()
        
        logger.debug(
            f"Totals recomputed for {self.order_id}: "
            f"subtotal=${self.subtotal:.2f}, tax=${self.tax:.2f}, "
            f"total=${self.total:.2f}"
        )
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate order for finalization.
        
        Returns:
            Validation result with is_valid flag and errors
        """
        self._transition(OrderState.VALIDATING)
        
        errors = []
        
        # Check items
        if len(self.items) == 0:
            errors.append("Order has no items")
        
        # Check total
        if self.total < self.min_order_total:
            errors.append(f"Order total below minimum: ${self.min_order_total:.2f}")
        
        # Check item validity
        for item in self.items.values():
            if item.quantity <= 0:
                errors.append(f"Invalid quantity for {item.name}")
            if item.price < 0:
                errors.append(f"Invalid price for {item.name}")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info(f"Order {self.order_id} validation passed")
            # Stay in VALIDATING state - finalize() will transition to FINALIZED
        else:
            logger.warning(f"Order {self.order_id} validation failed: {errors}")
            self._transition(OrderState.BUILDING)
        
        return {
            "is_valid": is_valid,
            "errors": errors,
            "order_id": self.order_id
        }
    
    # ========================================================================
    # FINALIZATION
    # ========================================================================
    
    def finalize(
        self,
        customer_phone: Optional[str] = None,
        customer_name: Optional[str] = None,
        delivery_address: Optional[str] = None,
        order_type: str = "pickup"
    ) -> Dict[str, Any]:
        """
        Finalize order (makes it immutable).
        
        Args:
            customer_phone: Customer phone number
            customer_name: Customer name
            delivery_address: Delivery address (if delivery)
            order_type: "pickup" or "delivery"
            
        Returns:
            Finalized order summary
        """
        if self._check_finalized():
            logger.warning(f"Order {self.order_id} already finalized")
            return self.get_summary()
        
        # Validate first
        validation = self.validate()
        if not validation["is_valid"]:
            logger.error(f"Cannot finalize invalid order: {validation['errors']}")
            return {
                "success": False,
                "errors": validation["errors"]
            }
        
        # Set customer info
        self.customer_phone = customer_phone
        self.customer_name = customer_name
        self.delivery_address = delivery_address
        self.order_type = order_type
        
        # Finalize
        self._transition(OrderState.FINALIZED)
        self._finalized = True
        self.finalized_at = datetime.utcnow()
        
        logger.info(
            f"Order {self.order_id} finalized: "
            f"{len(self.items)} items, total=${self.total:.2f}"
        )
        
        # Save to database
        self._save_to_db()
        
        return {
            "success": True,
            "order_id": self.order_id,
            "total": self.total,
            "summary": self.get_summary()
        }
    
    def cancel(self, reason: str = "Customer request"):
        """
        Cancel order.
        
        Args:
            reason: Cancellation reason
        """
        if self._check_finalized():
            return
        
        self._transition(OrderState.CANCELLED)
        logger.info(f"Order {self.order_id} cancelled: {reason}")
    
    # ========================================================================
    # GETTERS
    # ========================================================================
    
    def get_summary(self) -> Dict[str, Any]:
        """Get order summary."""
        return {
            "order_id": self.order_id,
            "call_id": self.call_id,
            "restaurant_id": self.restaurant_id,
            "state": self.state.value,
            "items": [asdict(item) for item in self.items.values()],
            "subtotal": self.subtotal,
            "tax": self.tax,
            "tax_rate": self.tax_rate,
            "total": self.total,
            "item_count": len(self.items),
            "customer_name": self.customer_name,
            "customer_phone": self.customer_phone,
            "delivery_address": self.delivery_address,
            "order_type": self.order_type,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "finalized_at": self.finalized_at.isoformat() if self.finalized_at else None,
            "is_finalized": self._finalized
        }
    
    def get_item_count(self) -> int:
        """Get total item count."""
        return len(self.items)
    
    def get_total_quantity(self) -> int:
        """Get total quantity across all items."""
        return sum(item.quantity for item in self.items.values())
    
    def has_items(self) -> bool:
        """Check if order has items."""
        return len(self.items) > 0
    
    def is_finalized(self) -> bool:
        """Check if order is finalized."""
        return self._finalized
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    def _save_to_db(self):
        """Save order to database."""
        try:
            from db import db
            
            order_data = {
                "order_id": self.order_id,
                "call_id": self.call_id,
                "restaurant_id": self.restaurant_id,
                "customer_phone": self.customer_phone,
                "customer_name": self.customer_name,
                "items": [asdict(item) for item in self.items.values()],
                "subtotal": self.subtotal,
                "tax": self.tax,
                "total": self.total,
                "order_type": self.order_type,
                "delivery_address": self.delivery_address,
                "status": self.state.value,
                "created_at": self.created_at.isoformat(),
                "finalized_at": self.finalized_at.isoformat() if self.finalized_at else None
            }
            
            db.store_order(order_data)
            logger.info(f"Order {self.order_id} saved to database")
        
        except Exception as e:
            logger.error(f"Failed to save order to database: {str(e)}")


# ============================================================================
# GLOBAL ORDER STORE
# ============================================================================

_order_store: Dict[str, Order] = {}


def create_order(call_id: str, restaurant_id: str) -> Order:
    """
    Create new order.
    
    Args:
        call_id: Call identifier
        restaurant_id: Restaurant identifier
        
    Returns:
        Order instance
    """
    if call_id in _order_store:
        logger.warning(f"Order already exists for call {call_id}, returning existing")
        return _order_store[call_id]
    
    order = Order(call_id, restaurant_id)
    _order_store[call_id] = order
    
    logger.info(f"Created order {order.order_id} for call {call_id}")
    return order


def get_order(call_id: str) -> Optional[Order]:
    """Get order for call."""
    return _order_store.get(call_id)


def clear_order(call_id: str):
    """Clear order (cleanup)."""
    order = _order_store.pop(call_id, None)
    if order:
        logger.info(f"Cleared order {order.order_id} for call {call_id}")


# Convenience wrapper functions
def add_item(call_id: str, item_id: str, name: str, price: float, quantity: int = 1, notes: Optional[str] = None) -> bool:
    """Add item to order."""
    order = get_order(call_id)
    return order.add_item(item_id, name, price, quantity, notes) if order else False


def remove_item(call_id: str, item_id: str) -> bool:
    """Remove item from order."""
    order = get_order(call_id)
    return order.remove_item(item_id) if order else False


def update_item_quantity(call_id: str, item_id: str, quantity: int) -> bool:
    """Update item quantity."""
    order = get_order(call_id)
    return order.update_item_quantity(item_id, quantity) if order else False


def calculate_total(call_id: str) -> float:
    """Get order total."""
    order = get_order(call_id)
    return order.total if order else 0.0


def validate_order(call_id: str) -> bool:
    """Validate order."""
    order = get_order(call_id)
    if not order:
        return False
    validation = order.validate()
    return validation["is_valid"]


def finalize_order(call_id: str, customer_phone: Optional[str] = None, customer_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Finalize order."""
    order = get_order(call_id)
    if not order:
        return None
    return order.finalize(customer_phone, customer_name)


def get_order_summary(call_id: str) -> Optional[Dict[str, Any]]:
    """Get order summary."""
    order = get_order(call_id)
    return order.get_summary() if order else None


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Order Module (Production)")
    print("=" * 50)
    
    # Create order
    call_id = "test_call_123"
    order = create_order(call_id, "rest_001")
    print(f"\nOrder created: {order.order_id}")
    print(f"State: {order.state.value}")
    
    # Add items
    order.add_item("item_1", "Large Pizza", 15.99, 2)
    order.add_item("item_2", "Chicken Wings", 9.99, 1)
    order.add_item("item_3", "Soda", 2.99, 3)
    
    print(f"\nItems added: {order.get_item_count()}")
    print(f"Total quantity: {order.get_total_quantity()}")
    print(f"Subtotal: ${order.subtotal:.2f}")
    print(f"Tax: ${order.tax:.2f}")
    print(f"Total: ${order.total:.2f}")
    
    # Update quantity
    order.update_item_quantity("item_1", 3)
    print(f"\nAfter update - Total: ${order.total:.2f}")
    
    # Validate
    validation = order.validate()
    print(f"\nValidation: {'PASS' if validation['is_valid'] else 'FAIL'}")
    
    # Finalize
    result = order.finalize(customer_phone="+1234567890", customer_name="John Doe")
    print(f"\nFinalized: {result['success']}")
    print(f"Order ID: {result['order_id']}")
    
    # Try to modify (should fail)
    success = order.add_item("item_4", "Dessert", 5.99, 1)
    print(f"\nTry to modify finalized order: {'Success' if success else 'Blocked (expected)'}")
    
    print("\n" + "=" * 50)
    print("Production order module ready")
