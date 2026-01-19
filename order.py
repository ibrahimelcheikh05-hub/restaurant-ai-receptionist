"""
Order Engine
============
Manage order creation, updates, totals, and validation.
Pure order management logic - no WebSocket, AI, or audio.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from memory import get_memory, create_call_memory, CallMemory
from db import db


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_TAX_RATE = 0.08  # 8% tax
MIN_ORDER_AMOUNT = 0.01  # Minimum order total
MAX_QUANTITY_PER_ITEM = 99


# ============================================================================
# ORDER CREATION
# ============================================================================

def create_order(call_id: str, restaurant_id: str) -> Dict[str, Any]:
    """
    Create a new order for a call.
    Initializes memory with empty order state.
    
    Args:
        call_id: Unique call identifier
        restaurant_id: Restaurant identifier
        
    Returns:
        Dictionary with order initialization status
        
    Raises:
        ValueError: If call_id or restaurant_id is empty
        RuntimeError: If order already exists for this call
        
    Example:
        >>> order = create_order("CA123", "rest_123")
        >>> print(order['status'])  # 'created'
    """
    # Validate input
    if not call_id or call_id.strip() == "":
        raise ValueError("call_id cannot be empty")
    
    if not restaurant_id or restaurant_id.strip() == "":
        raise ValueError("restaurant_id cannot be empty")
    
    # Check if memory already exists
    memory = get_memory(call_id)
    if memory is not None:
        raise RuntimeError(
            f"Order already exists for call_id '{call_id}'. "
            f"Use add_item() to modify existing order."
        )
    
    # Create new memory
    memory = create_call_memory(call_id)
    memory.set_restaurant_id(restaurant_id)
    memory.set_state("order_active")
    
    return {
        "call_id": call_id,
        "restaurant_id": restaurant_id,
        "status": "created",
        "items": [],
        "total": 0.0,
        "created_at": datetime.utcnow().isoformat()
    }


# ============================================================================
# ORDER ITEM MANAGEMENT
# ============================================================================

def add_item(
    call_id: str,
    item: Dict[str, Any],
    quantity: int = 1,
    customizations: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Add an item to the order.
    
    Args:
        call_id: Call identifier
        item: Menu item dictionary with id, name, price
        quantity: Quantity to add (default: 1)
        customizations: Optional list of customizations/modifications
        
    Returns:
        Updated order summary
        
    Raises:
        ValueError: If call_id is empty, item invalid, or quantity invalid
        RuntimeError: If order not found
        
    Example:
        >>> item = {"id": "pizza_1", "name": "Margherita", "price": 12.99}
        >>> result = add_item("CA123", item, quantity=2, customizations=["extra cheese"])
    """
    # Validate input
    if not call_id or call_id.strip() == "":
        raise ValueError("call_id cannot be empty")
    
    if not item or not isinstance(item, dict):
        raise ValueError("item must be a dictionary")
    
    # Validate item structure
    required_fields = ["id", "name", "price"]
    for field in required_fields:
        if field not in item:
            raise ValueError(f"item missing required field: '{field}'")
    
    # Validate quantity
    if quantity < 1:
        raise ValueError(f"quantity must be at least 1, got {quantity}")
    
    if quantity > MAX_QUANTITY_PER_ITEM:
        raise ValueError(
            f"quantity cannot exceed {MAX_QUANTITY_PER_ITEM}, got {quantity}"
        )
    
    # Get memory
    memory = get_memory(call_id)
    if memory is None:
        raise RuntimeError(
            f"No order found for call_id '{call_id}'. "
            f"Call create_order() first."
        )
    
    # Extract item details
    item_id = item["id"]
    name = item["name"]
    price = float(item["price"])
    
    # Validate price
    if price < 0:
        raise ValueError(f"item price cannot be negative: ${price}")
    
    # Check if item already exists in order
    existing_item = _find_order_item(memory, item_id)
    
    if existing_item:
        # Update quantity of existing item
        new_quantity = existing_item["quantity"] + quantity
        if new_quantity > MAX_QUANTITY_PER_ITEM:
            raise ValueError(
                f"Total quantity for '{name}' would exceed {MAX_QUANTITY_PER_ITEM}"
            )
        memory.update_order_item_quantity(item_id, new_quantity)
    else:
        # Add new item
        memory.add_order_item(
            item_id=item_id,
            name=name,
            quantity=quantity,
            price=price,
            customizations=customizations or []
        )
    
    # Return updated order summary
    return get_order_summary(call_id)


def remove_item(call_id: str, item_id: str) -> Dict[str, Any]:
    """
    Remove an item from the order.
    
    Args:
        call_id: Call identifier
        item_id: ID of item to remove
        
    Returns:
        Updated order summary
        
    Raises:
        ValueError: If call_id or item_id is empty
        RuntimeError: If order not found
        
    Example:
        >>> result = remove_item("CA123", "pizza_1")
    """
    # Validate input
    if not call_id or call_id.strip() == "":
        raise ValueError("call_id cannot be empty")
    
    if not item_id or item_id.strip() == "":
        raise ValueError("item_id cannot be empty")
    
    # Get memory
    memory = get_memory(call_id)
    if memory is None:
        raise RuntimeError(
            f"No order found for call_id '{call_id}'"
        )
    
    # Remove item
    removed = memory.remove_order_item(item_id)
    
    if not removed:
        # Item not found - not necessarily an error, just return current state
        pass
    
    # Return updated order summary
    return get_order_summary(call_id)


def update_item_quantity(
    call_id: str,
    item_id: str,
    quantity: int
) -> Dict[str, Any]:
    """
    Update quantity of an item in the order.
    
    Args:
        call_id: Call identifier
        item_id: ID of item to update
        quantity: New quantity (0 to remove item)
        
    Returns:
        Updated order summary
        
    Raises:
        ValueError: If call_id or item_id is empty, or quantity invalid
        RuntimeError: If order not found
        
    Example:
        >>> result = update_item_quantity("CA123", "pizza_1", 3)
    """
    # Validate input
    if not call_id or call_id.strip() == "":
        raise ValueError("call_id cannot be empty")
    
    if not item_id or item_id.strip() == "":
        raise ValueError("item_id cannot be empty")
    
    if quantity < 0:
        raise ValueError(f"quantity cannot be negative, got {quantity}")
    
    if quantity > MAX_QUANTITY_PER_ITEM:
        raise ValueError(
            f"quantity cannot exceed {MAX_QUANTITY_PER_ITEM}, got {quantity}"
        )
    
    # Get memory
    memory = get_memory(call_id)
    if memory is None:
        raise RuntimeError(
            f"No order found for call_id '{call_id}'"
        )
    
    # If quantity is 0, remove the item
    if quantity == 0:
        memory.remove_order_item(item_id)
    else:
        # Update quantity
        updated = memory.update_order_item_quantity(item_id, quantity)
        if not updated:
            raise ValueError(f"Item '{item_id}' not found in order")
    
    # Return updated order summary
    return get_order_summary(call_id)


def clear_order(call_id: str) -> Dict[str, Any]:
    """
    Clear all items from the order.
    
    Args:
        call_id: Call identifier
        
    Returns:
        Empty order summary
        
    Raises:
        ValueError: If call_id is empty
        RuntimeError: If order not found
        
    Example:
        >>> result = clear_order("CA123")
    """
    # Validate input
    if not call_id or call_id.strip() == "":
        raise ValueError("call_id cannot be empty")
    
    # Get memory
    memory = get_memory(call_id)
    if memory is None:
        raise RuntimeError(
            f"No order found for call_id '{call_id}'"
        )
    
    # Clear order
    memory.clear_order()
    
    # Return empty order summary
    return get_order_summary(call_id)


# ============================================================================
# ORDER CALCULATION
# ============================================================================

def calculate_total(
    call_id: str,
    tax_rate: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate order totals (subtotal, tax, total).
    
    Args:
        call_id: Call identifier
        tax_rate: Optional custom tax rate (default: 8%)
        
    Returns:
        Dictionary with:
        - subtotal: Sum of item prices
        - tax: Tax amount
        - total: Final total
        - tax_rate: Tax rate used
        
    Raises:
        ValueError: If call_id is empty or tax_rate invalid
        RuntimeError: If order not found
        
    Example:
        >>> totals = calculate_total("CA123")
        >>> print(f"Total: ${totals['total']:.2f}")
    """
    # Validate input
    if not call_id or call_id.strip() == "":
        raise ValueError("call_id cannot be empty")
    
    # Validate tax rate if provided
    if tax_rate is not None:
        if tax_rate < 0 or tax_rate > 1:
            raise ValueError(
                f"tax_rate must be between 0 and 1, got {tax_rate}"
            )
    else:
        tax_rate = DEFAULT_TAX_RATE
    
    # Get memory
    memory = get_memory(call_id)
    if memory is None:
        raise RuntimeError(
            f"No order found for call_id '{call_id}'"
        )
    
    # Get order summary from memory
    order_summary = memory.get_order_summary()
    
    # Calculate subtotal
    subtotal = sum(
        item["price"] * item["quantity"]
        for item in order_summary["items"]
    )
    
    # Calculate tax
    tax = subtotal * tax_rate
    
    # Calculate total
    total = subtotal + tax
    
    return {
        "subtotal": round(subtotal, 2),
        "tax": round(tax, 2),
        "total": round(total, 2),
        "tax_rate": tax_rate
    }


# ============================================================================
# ORDER SUMMARY
# ============================================================================

def get_order_summary(call_id: str) -> Dict[str, Any]:
    """
    Get complete order summary.
    
    Args:
        call_id: Call identifier
        
    Returns:
        Dictionary with complete order details
        
    Raises:
        ValueError: If call_id is empty
        RuntimeError: If order not found
        
    Example:
        >>> summary = get_order_summary("CA123")
        >>> print(f"Items: {summary['item_count']}")
    """
    # Validate input
    if not call_id or call_id.strip() == "":
        raise ValueError("call_id cannot be empty")
    
    # Get memory
    memory = get_memory(call_id)
    if memory is None:
        raise RuntimeError(
            f"No order found for call_id '{call_id}'"
        )
    
    # Get order from memory
    order = memory.get_order_summary()
    
    # Calculate totals
    totals = calculate_total(call_id)
    
    # Build comprehensive summary
    return {
        "call_id": call_id,
        "restaurant_id": memory.restaurant_id,
        "items": order["items"],
        "item_count": len(order["items"]),
        "subtotal": totals["subtotal"],
        "tax": totals["tax"],
        "total": totals["total"],
        "tax_rate": totals["tax_rate"],
        "is_empty": len(order["items"]) == 0
    }


# ============================================================================
# ORDER VALIDATION
# ============================================================================

def validate_order(call_id: str) -> Tuple[bool, List[str]]:
    """
    Validate order before finalization.
    
    Args:
        call_id: Call identifier
        
    Returns:
        Tuple of (is_valid, list_of_errors)
        
    Example:
        >>> valid, errors = validate_order("CA123")
        >>> if not valid:
        >>>     print(f"Errors: {errors}")
    """
    errors = []
    
    try:
        # Get order summary
        summary = get_order_summary(call_id)
        
        # Check if order is empty
        if summary["is_empty"]:
            errors.append("Order is empty - no items added")
        
        # Check minimum order amount
        if summary["total"] < MIN_ORDER_AMOUNT:
            errors.append(
                f"Order total (${summary['total']:.2f}) is below "
                f"minimum (${MIN_ORDER_AMOUNT:.2f})"
            )
        
        # Validate each item
        for item in summary["items"]:
            # Check quantity
            if item["quantity"] < 1:
                errors.append(
                    f"Item '{item['name']}' has invalid quantity: {item['quantity']}"
                )
            
            # Check price
            if item["price"] < 0:
                errors.append(
                    f"Item '{item['name']}' has invalid price: ${item['price']}"
                )
        
        # Get memory for additional validation
        memory = get_memory(call_id)
        if memory:
            # Check if restaurant ID is set
            if not memory.restaurant_id:
                errors.append("Restaurant ID not set")
            
            # Check if customer info is available (optional but recommended)
            if not memory.customer_phone:
                errors.append("Customer phone number not set (recommended)")
    
    except Exception as e:
        errors.append(f"Validation error: {str(e)}")
    
    return len(errors) == 0, errors


# ============================================================================
# ORDER FINALIZATION
# ============================================================================

def finalize_order(
    call_id: str,
    customer_phone: Optional[str] = None,
    customer_name: Optional[str] = None,
    delivery_address: Optional[str] = None,
    special_instructions: Optional[str] = None
) -> Dict[str, Any]:
    """
    Finalize order and persist to database.
    
    Args:
        call_id: Call identifier
        customer_phone: Customer phone number
        customer_name: Customer name
        delivery_address: Delivery address (if applicable)
        special_instructions: Any special instructions
        
    Returns:
        Finalized order with database ID
        
    Raises:
        ValueError: If order validation fails
        RuntimeError: If database save fails
        
    Example:
        >>> final = finalize_order(
        >>>     "CA123",
        >>>     customer_phone="+1234567890",
        >>>     customer_name="John Doe"
        >>> )
        >>> print(f"Order ID: {final['order_id']}")
    """
    # Get memory
    memory = get_memory(call_id)
    if memory is None:
        raise RuntimeError(
            f"No order found for call_id '{call_id}'"
        )
    
    # Update customer info if provided
    if customer_phone or customer_name:
        memory.set_customer_info(phone=customer_phone, name=customer_name)
    
    # Validate order
    is_valid, errors = validate_order(call_id)
    if not is_valid:
        raise ValueError(
            f"Order validation failed: {'; '.join(errors)}"
        )
    
    # Get order summary
    summary = get_order_summary(call_id)
    
    # Prepare order data for database
    order_data = {
        "restaurant_id": summary["restaurant_id"],
        "customer_phone": customer_phone or memory.customer_phone or "",
        "customer_name": customer_name or memory.customer_name,
        "items": summary["items"],
        "total_amount": summary["total"],
        "status": "pending",
        "delivery_address": delivery_address,
        "special_instructions": special_instructions
    }
    
    try:
        # Save to database
        saved_order = db.store_order(order_data)
        
        # Update memory state
        memory.set_state("order_finalized")
        
        # Return finalized order
        return {
            "order_id": saved_order.get("id"),
            "call_id": call_id,
            "restaurant_id": summary["restaurant_id"],
            "customer_phone": order_data["customer_phone"],
            "customer_name": order_data["customer_name"],
            "items": summary["items"],
            "item_count": summary["item_count"],
            "subtotal": summary["subtotal"],
            "tax": summary["tax"],
            "total": summary["total"],
            "status": "pending",
            "delivery_address": delivery_address,
            "special_instructions": special_instructions,
            "created_at": saved_order.get("created_at"),
            "finalized": True
        }
        
    except Exception as e:
        raise RuntimeError(f"Failed to save order to database: {str(e)}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _find_order_item(
    memory: CallMemory,
    item_id: str
) -> Optional[Dict[str, Any]]:
    """
    Find an item in the order by ID.
    
    Args:
        memory: CallMemory instance
        item_id: Item identifier
        
    Returns:
        Item dictionary or None if not found
    """
    order = memory.get_order_summary()
    for item in order["items"]:
        if item["item_id"] == item_id:
            return item
    return None


def get_order_items(call_id: str) -> List[Dict[str, Any]]:
    """
    Get list of items in the order.
    
    Args:
        call_id: Call identifier
        
    Returns:
        List of order items
        
    Example:
        >>> items = get_order_items("CA123")
        >>> for item in items:
        >>>     print(f"{item['name']}: ${item['price']}")
    """
    summary = get_order_summary(call_id)
    return summary["items"]


def get_item_count(call_id: str) -> int:
    """
    Get total number of items in order.
    
    Args:
        call_id: Call identifier
        
    Returns:
        Total item count
        
    Example:
        >>> count = get_item_count("CA123")
        >>> print(f"Order has {count} items")
    """
    summary = get_order_summary(call_id)
    return summary["item_count"]


def is_order_empty(call_id: str) -> bool:
    """
    Check if order is empty.
    
    Args:
        call_id: Call identifier
        
    Returns:
        True if order has no items
        
    Example:
        >>> if is_order_empty("CA123"):
        >>>     print("Please add items to your order")
    """
    summary = get_order_summary(call_id)
    return summary["is_empty"]


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the order engine.
    """
    
    print("Order Engine Example")
    print("=" * 50)
    
    # Simulate a call
    call_id = "CA123test"
    restaurant_id = "rest_123"
    
    try:
        # Create order
        print("\n1. Creating order...")
        order = create_order(call_id, restaurant_id)
        print(f"   ✓ Order created for call: {call_id}")
        
        # Add items
        print("\n2. Adding items...")
        
        item1 = {
            "id": "pizza_1",
            "name": "Margherita Pizza",
            "price": 12.99
        }
        add_item(call_id, item1, quantity=2, customizations=["extra cheese"])
        print(f"   ✓ Added 2x {item1['name']}")
        
        item2 = {
            "id": "drink_1",
            "name": "Coca-Cola",
            "price": 2.50
        }
        add_item(call_id, item2, quantity=3)
        print(f"   ✓ Added 3x {item2['name']}")
        
        # Get summary
        print("\n3. Order Summary:")
        summary = get_order_summary(call_id)
        print(f"   Items: {summary['item_count']}")
        for item in summary['items']:
            print(f"   - {item['quantity']}x {item['name']} @ ${item['price']}")
        
        # Calculate totals
        print("\n4. Totals:")
        totals = calculate_total(call_id)
        print(f"   Subtotal: ${totals['subtotal']:.2f}")
        print(f"   Tax ({totals['tax_rate']*100:.0f}%): ${totals['tax']:.2f}")
        print(f"   Total: ${totals['total']:.2f}")
        
        # Update quantity
        print("\n5. Updating quantity...")
        update_item_quantity(call_id, "drink_1", 2)
        print("   ✓ Updated Coca-Cola quantity to 2")
        
        # New totals
        new_totals = calculate_total(call_id)
        print(f"   New total: ${new_totals['total']:.2f}")
        
        # Validate
        print("\n6. Validating order...")
        is_valid, errors = validate_order(call_id)
        if is_valid:
            print("   ✓ Order is valid")
        else:
            print("   ✗ Validation errors:")
            for error in errors:
                print(f"     - {error}")
        
        # Note: Finalization would require actual database
        print("\n7. Finalization:")
        print("   (Skipped - requires database connection)")
        print("   In production:")
        print("   final = finalize_order(call_id,")
        print("       customer_phone='+1234567890',")
        print("       customer_name='John Doe')")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
    
    finally:
        # Cleanup
        from memory import clear_memory
        clear_memory(call_id)
        print("\n✓ Test cleanup complete")
