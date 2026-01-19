"""
Upsell Intelligence Layer
==========================
Smart upsell and cross-sell recommendations.
Pure logic-based system - no AI, no database writes.
"""

from typing import Dict, List, Any, Optional, Set
from collections import Counter


# ============================================================================
# CONFIGURATION
# ============================================================================

# Common item associations for cross-selling
CROSS_SELL_RULES = {
    "pizza": ["drinks", "sides", "desserts"],
    "burger": ["fries", "drinks", "sides"],
    "pasta": ["bread", "salad", "drinks"],
    "sandwich": ["chips", "drinks", "sides"],
    "salad": ["drinks", "bread", "protein"],
    "entree": ["appetizers", "sides", "drinks", "desserts"]
}

# Price-based upsell thresholds
UPSELL_PRICE_MARGIN = 1.5  # Suggest items up to 1.5x current item price
UPSELL_MAX_SUGGESTIONS = 5
CROSS_SELL_MAX_SUGGESTIONS = 3

# Minimum order value for premium upsells
PREMIUM_UPSELL_THRESHOLD = 20.00


# ============================================================================
# CORE UPSELL FUNCTION
# ============================================================================

def suggest_upsells(
    menu: Dict[str, Any],
    current_order: List[Dict[str, Any]],
    max_suggestions: int = UPSELL_MAX_SUGGESTIONS
) -> List[Dict[str, Any]]:
    """
    Suggest upsell and cross-sell items based on current order.
    
    Args:
        menu: Menu dictionary from menu.py (with 'items' key)
        current_order: List of items currently in order
        max_suggestions: Maximum number of suggestions to return
        
    Returns:
        List of suggested items, ranked by relevance
        Each suggestion includes:
        - item: Menu item dictionary
        - reason: Why this is suggested
        - type: 'upsell' or 'cross-sell'
        - score: Relevance score (higher = more relevant)
        
    Example:
        >>> suggestions = suggest_upsells(menu, current_order)
        >>> for s in suggestions:
        >>>     print(f"{s['item']['name']}: {s['reason']}")
    """
    # Validate input
    if not menu or not menu.get("items"):
        return []
    
    if not current_order:
        # No items yet - suggest popular or featured items
        return _suggest_initial_items(menu, max_suggestions)
    
    # Get all menu items
    all_items = menu["items"]
    
    # Get items already in order (to avoid duplicates)
    order_item_ids = {item["item_id"] for item in current_order}
    
    # Calculate order statistics
    order_stats = _calculate_order_stats(current_order)
    
    # Generate suggestions
    suggestions = []
    
    # 1. Cross-sell suggestions (complementary items)
    cross_sells = _generate_cross_sell_suggestions(
        all_items,
        current_order,
        order_item_ids,
        order_stats
    )
    suggestions.extend(cross_sells)
    
    # 2. Upsell suggestions (premium versions)
    upsells = _generate_upsell_suggestions(
        all_items,
        current_order,
        order_item_ids,
        order_stats
    )
    suggestions.extend(upsells)
    
    # 3. Category completion (missing categories)
    category_suggestions = _generate_category_suggestions(
        all_items,
        current_order,
        order_item_ids,
        order_stats
    )
    suggestions.extend(category_suggestions)
    
    # 4. Value-based suggestions (combos, deals)
    value_suggestions = _generate_value_suggestions(
        all_items,
        current_order,
        order_item_ids,
        order_stats
    )
    suggestions.extend(value_suggestions)
    
    # Rank suggestions by score
    suggestions.sort(key=lambda x: x["score"], reverse=True)
    
    # Return top suggestions
    return suggestions[:max_suggestions]


# ============================================================================
# SUGGESTION GENERATORS
# ============================================================================

def _generate_cross_sell_suggestions(
    all_items: List[Dict[str, Any]],
    current_order: List[Dict[str, Any]],
    order_item_ids: Set[str],
    order_stats: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Generate cross-sell suggestions (complementary items).
    """
    suggestions = []
    
    # Get categories in current order
    order_categories = {
        item.get("category", "").lower()
        for item in current_order
    }
    
    # Determine what categories to suggest
    suggested_categories = set()
    for order_cat in order_categories:
        # Check cross-sell rules
        for rule_key, cross_sell_cats in CROSS_SELL_RULES.items():
            if rule_key in order_cat:
                suggested_categories.update(cross_sell_cats)
    
    # Find items in suggested categories
    for item in all_items:
        if item["id"] in order_item_ids:
            continue  # Skip items already in order
        
        item_category = item.get("category", "").lower()
        
        # Check if item matches suggested categories
        for suggested_cat in suggested_categories:
            if suggested_cat in item_category:
                suggestions.append({
                    "item": item,
                    "reason": f"Pairs well with your {list(order_categories)[0]}",
                    "type": "cross-sell",
                    "score": 8.0  # High relevance
                })
                break
    
    return suggestions


def _generate_upsell_suggestions(
    all_items: List[Dict[str, Any]],
    current_order: List[Dict[str, Any]],
    order_item_ids: Set[str],
    order_stats: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Generate upsell suggestions (premium versions of current items).
    """
    suggestions = []
    
    # For each item in order, find premium alternatives
    for order_item in current_order:
        order_item_name = order_item.get("name", "").lower()
        order_item_category = order_item.get("category", "").lower()
        order_item_price = order_item.get("price", 0.0)
        
        # Find items in same category with higher price
        for menu_item in all_items:
            if menu_item["id"] in order_item_ids:
                continue
            
            menu_category = menu_item.get("category", "").lower()
            menu_price = menu_item.get("price", 0.0)
            
            # Must be same category
            if menu_category != order_item_category:
                continue
            
            # Must be more expensive (but not too much)
            if menu_price <= order_item_price:
                continue
            
            if menu_price > order_item_price * UPSELL_PRICE_MARGIN:
                continue
            
            # Calculate price difference
            price_diff = menu_price - order_item_price
            
            suggestions.append({
                "item": menu_item,
                "reason": f"Premium upgrade (+${price_diff:.2f})",
                "type": "upsell",
                "score": 7.0  # Good relevance
            })
    
    return suggestions


def _generate_category_suggestions(
    all_items: List[Dict[str, Any]],
    current_order: List[Dict[str, Any]],
    order_item_ids: Set[str],
    order_stats: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Suggest items from missing categories to complete the meal.
    """
    suggestions = []
    
    # Categories in order
    order_categories = {
        item.get("category", "").lower()
        for item in current_order
    }
    
    # All available categories
    all_categories = {
        item.get("category", "").lower()
        for item in all_items
    }
    
    # Missing categories
    missing_categories = all_categories - order_categories
    
    # Common meal completion categories
    completion_categories = ["drinks", "desserts", "sides", "appetizers"]
    
    for cat in completion_categories:
        if any(cat in missing_cat for missing_cat in missing_categories):
            # Find items in this category
            for item in all_items:
                if item["id"] in order_item_ids:
                    continue
                
                item_cat = item.get("category", "").lower()
                if cat in item_cat:
                    suggestions.append({
                        "item": item,
                        "reason": f"Complete your meal with a {cat.rstrip('s')}",
                        "type": "cross-sell",
                        "score": 6.0  # Moderate relevance
                    })
                    break  # One per category
    
    return suggestions


def _generate_value_suggestions(
    all_items: List[Dict[str, Any]],
    current_order: List[Dict[str, Any]],
    order_item_ids: Set[str],
    order_stats: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Suggest value items (affordable additions).
    """
    suggestions = []
    
    # Order total
    order_total = order_stats["total_value"]
    
    # If order is small, suggest affordable items
    if order_total < PREMIUM_UPSELL_THRESHOLD:
        # Find low-price items
        affordable_items = [
            item for item in all_items
            if item["id"] not in order_item_ids
            and item.get("price", 999) < 5.00
        ]
        
        # Sort by price (cheapest first)
        affordable_items.sort(key=lambda x: x.get("price", 0))
        
        # Suggest top 2 affordable items
        for item in affordable_items[:2]:
            suggestions.append({
                "item": item,
                "reason": "Great value addition",
                "type": "cross-sell",
                "score": 5.0  # Lower relevance
            })
    
    return suggestions


# ============================================================================
# ORDER ANALYSIS
# ============================================================================

def _calculate_order_stats(current_order: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics about the current order.
    
    Args:
        current_order: List of items in order
        
    Returns:
        Dictionary with order statistics
    """
    total_items = sum(item.get("quantity", 1) for item in current_order)
    total_value = sum(
        item.get("price", 0.0) * item.get("quantity", 1)
        for item in current_order
    )
    
    categories = [item.get("category", "Other") for item in current_order]
    category_counts = Counter(categories)
    
    avg_item_price = total_value / total_items if total_items > 0 else 0.0
    
    return {
        "total_items": total_items,
        "total_value": total_value,
        "unique_items": len(current_order),
        "categories": list(category_counts.keys()),
        "category_counts": dict(category_counts),
        "avg_item_price": avg_item_price
    }


def _suggest_initial_items(
    menu: Dict[str, Any],
    max_suggestions: int
) -> List[Dict[str, Any]]:
    """
    Suggest items when order is empty (initial suggestions).
    
    Args:
        menu: Menu dictionary
        max_suggestions: Max number to return
        
    Returns:
        List of suggested items
    """
    suggestions = []
    all_items = menu.get("items", [])
    
    if not all_items:
        return []
    
    # Strategy: Suggest popular categories and mid-priced items
    
    # 1. Find items from main categories
    main_categories = ["pizza", "burger", "pasta", "entree", "main"]
    
    for category in main_categories:
        for item in all_items:
            item_cat = item.get("category", "").lower()
            if category in item_cat:
                suggestions.append({
                    "item": item,
                    "reason": "Popular choice",
                    "type": "featured",
                    "score": 7.0
                })
                break  # One per category
    
    # 2. Add some variety from other categories
    other_items = [
        item for item in all_items
        if item not in [s["item"] for s in suggestions]
    ]
    
    # Sort by price (mid-range first)
    if other_items:
        prices = [item.get("price", 0.0) for item in other_items]
        avg_price = sum(prices) / len(prices) if prices else 0.0
        
        other_items.sort(
            key=lambda x: abs(x.get("price", 0.0) - avg_price)
        )
        
        for item in other_items[:2]:
            suggestions.append({
                "item": item,
                "reason": "Try this",
                "type": "featured",
                "score": 5.0
            })
    
    return suggestions[:max_suggestions]


# ============================================================================
# ADVANCED UPSELL STRATEGIES
# ============================================================================

def suggest_bundle_deals(
    menu: Dict[str, Any],
    current_order: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Suggest bundle or combo deals.
    
    Args:
        menu: Menu dictionary
        current_order: Current order items
        
    Returns:
        List of bundle suggestions
        
    Example:
        >>> bundles = suggest_bundle_deals(menu, current_order)
    """
    # This is a placeholder for bundle logic
    # In production, you'd have bundle definitions in your database
    bundles = []
    
    # Example: If customer has main item but no drink, suggest combo
    has_main = any(
        "pizza" in item.get("category", "").lower() or
        "burger" in item.get("category", "").lower()
        for item in current_order
    )
    
    has_drink = any(
        "drink" in item.get("category", "").lower()
        for item in current_order
    )
    
    if has_main and not has_drink:
        bundles.append({
            "suggestion": "Add a drink for just $2 more",
            "type": "combo",
            "score": 9.0
        })
    
    return bundles


def suggest_quantity_upsell(
    current_order: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Suggest increasing quantity of existing items.
    
    Args:
        current_order: Current order items
        
    Returns:
        List of quantity upsell suggestions
        
    Example:
        >>> qty_upsells = suggest_quantity_upsell(current_order)
    """
    suggestions = []
    
    # Look for items with quantity = 1
    for item in current_order:
        if item.get("quantity", 0) == 1:
            item_name = item.get("name", "item")
            suggestions.append({
                "item": item,
                "suggestion": f"Add another {item_name}?",
                "reason": "Great value when you buy more",
                "type": "quantity-upsell",
                "score": 6.0
            })
    
    return suggestions


def suggest_by_time_of_day(
    menu: Dict[str, Any],
    hour: int
) -> List[Dict[str, Any]]:
    """
    Suggest items based on time of day.
    
    Args:
        menu: Menu dictionary
        hour: Hour of day (0-23)
        
    Returns:
        List of time-appropriate suggestions
        
    Example:
        >>> from datetime import datetime
        >>> hour = datetime.now().hour
        >>> suggestions = suggest_by_time_of_day(menu, hour)
    """
    suggestions = []
    all_items = menu.get("items", [])
    
    # Breakfast time (6-11)
    if 6 <= hour < 11:
        breakfast_categories = ["breakfast", "coffee", "pastry"]
        for item in all_items:
            cat = item.get("category", "").lower()
            if any(b in cat for b in breakfast_categories):
                suggestions.append({
                    "item": item,
                    "reason": "Perfect for breakfast",
                    "type": "time-based",
                    "score": 8.0
                })
    
    # Lunch time (11-15)
    elif 11 <= hour < 15:
        lunch_categories = ["sandwich", "salad", "soup"]
        for item in all_items:
            cat = item.get("category", "").lower()
            if any(l in cat for l in lunch_categories):
                suggestions.append({
                    "item": item,
                    "reason": "Great lunch option",
                    "type": "time-based",
                    "score": 8.0
                })
    
    # Dinner time (17-22)
    elif 17 <= hour < 22:
        dinner_categories = ["entree", "pasta", "pizza", "main"]
        for item in all_items:
            cat = item.get("category", "").lower()
            if any(d in cat for d in dinner_categories):
                suggestions.append({
                    "item": item,
                    "reason": "Perfect for dinner",
                    "type": "time-based",
                    "score": 8.0
                })
    
    # Late night (22-6)
    else:
        late_categories = ["snack", "dessert"]
        for item in all_items:
            cat = item.get("category", "").lower()
            if any(ln in cat for ln in late_categories):
                suggestions.append({
                    "item": item,
                    "reason": "Late night snack",
                    "type": "time-based",
                    "score": 7.0
                })
    
    return suggestions


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_suggestion_text(suggestion: Dict[str, Any]) -> str:
    """
    Format a suggestion into natural language text.
    
    Args:
        suggestion: Suggestion dictionary
        
    Returns:
        Natural language suggestion text
        
    Example:
        >>> text = format_suggestion_text(suggestion)
        >>> print(text)  # "Would you like to add Coca-Cola? Pairs well with your pizza."
    """
    item = suggestion["item"]
    reason = suggestion["reason"]
    item_name = item.get("name", "this item")
    item_price = item.get("price", 0.0)
    
    return f"Would you like to add {item_name} (${item_price:.2f})? {reason}."


def get_top_suggestion(
    menu: Dict[str, Any],
    current_order: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Get the single best suggestion.
    
    Args:
        menu: Menu dictionary
        current_order: Current order
        
    Returns:
        Top suggestion or None
        
    Example:
        >>> top = get_top_suggestion(menu, current_order)
        >>> if top:
        >>>     print(format_suggestion_text(top))
    """
    suggestions = suggest_upsells(menu, current_order, max_suggestions=1)
    return suggestions[0] if suggestions else None


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the upsell engine.
    """
    
    print("Upsell Intelligence Example")
    print("=" * 50)
    
    # Mock menu
    menu = {
        "items": [
            {"id": "pizza_1", "name": "Margherita Pizza", "price": 12.99, "category": "Pizza"},
            {"id": "pizza_2", "name": "Deluxe Pizza", "price": 16.99, "category": "Pizza"},
            {"id": "drink_1", "name": "Coca-Cola", "price": 2.50, "category": "Drinks"},
            {"id": "side_1", "name": "Garlic Bread", "price": 4.99, "category": "Sides"},
            {"id": "dessert_1", "name": "Tiramisu", "price": 6.99, "category": "Desserts"},
            {"id": "salad_1", "name": "Caesar Salad", "price": 8.99, "category": "Salads"},
        ]
    }
    
    # Mock current order
    current_order = [
        {
            "item_id": "pizza_1",
            "name": "Margherita Pizza",
            "price": 12.99,
            "quantity": 1,
            "category": "Pizza"
        }
    ]
    
    print("\n1. Current Order:")
    for item in current_order:
        print(f"   - {item['quantity']}x {item['name']} (${item['price']})")
    
    # Get suggestions
    print("\n2. Upsell Suggestions:")
    suggestions = suggest_upsells(menu, current_order)
    
    for i, suggestion in enumerate(suggestions, 1):
        item = suggestion["item"]
        print(f"   {i}. {item['name']} (${item['price']:.2f})")
        print(f"      Type: {suggestion['type']}")
        print(f"      Reason: {suggestion['reason']}")
        print(f"      Score: {suggestion['score']}")
        print()
    
    # Format as natural language
    print("3. Natural Language Suggestions:")
    for suggestion in suggestions[:3]:
        text = format_suggestion_text(suggestion)
        print(f"   - {text}")
    
    # Top suggestion
    print("\n4. Top Recommendation:")
    top = get_top_suggestion(menu, current_order)
    if top:
        print(f"   {format_suggestion_text(top)}")
    
    # Empty order suggestions
    print("\n5. Initial Suggestions (Empty Order):")
    initial_suggestions = suggest_upsells(menu, [])
    for suggestion in initial_suggestions[:3]:
        item = suggestion["item"]
        print(f"   - {item['name']}: {suggestion['reason']}")
