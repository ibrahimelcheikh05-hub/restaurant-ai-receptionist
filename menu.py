"""
Menu Access Module
==================
Fetch and format restaurant menus for AI voice agent usage.
Pure data access and formatting - no AI, no audio, no WebSocket logic.
"""

from typing import Dict, List, Any, Optional
from db import db


# ============================================================================
# MENU RETRIEVAL
# ============================================================================

def get_menu(restaurant_id: str) -> Dict[str, Any]:
    """
    Fetch restaurant menu from database.
    
    Args:
        restaurant_id: Unique identifier for the restaurant
        
    Returns:
        Dictionary containing:
        - items: List of menu items
        - categories: List of unique categories
        - item_count: Total number of items
        - has_items: Boolean indicating if menu has items
        
    Raises:
        ValueError: If restaurant_id is empty
        Exception: If database fetch fails
        
    Example:
        >>> menu = get_menu("rest_123")
        >>> print(f"Menu has {menu['item_count']} items")
    """
    # Validate input
    if not restaurant_id or restaurant_id.strip() == "":
        raise ValueError("restaurant_id cannot be empty")
    
    try:
        # Fetch menu items from database
        items = db.fetch_menu(restaurant_id)
        
        # Extract unique categories
        categories = list(set(
            item.get("category", "Other")
            for item in items
            if item.get("category")
        ))
        categories.sort()  # Alphabetical order
        
        # Build structured response
        return {
            "items": items,
            "categories": categories,
            "item_count": len(items),
            "has_items": len(items) > 0
        }
        
    except Exception as e:
        raise Exception(f"Failed to fetch menu for restaurant '{restaurant_id}': {str(e)}")


# ============================================================================
# MENU FORMATTING FOR AI
# ============================================================================

def format_menu_for_prompt(menu: Dict[str, Any]) -> str:
    """
    Format menu into a clean string for AI prompt inclusion.
    Optimized for LLM context windows and natural conversation.
    
    Args:
        menu: Menu dictionary from get_menu()
        
    Returns:
        Formatted menu string ready for AI prompt
        
    Example:
        >>> menu = get_menu("rest_123")
        >>> prompt_text = format_menu_for_prompt(menu)
        >>> print(prompt_text)
    """
    # Handle empty menu
    if not menu.get("has_items", False):
        return "No menu items available. Please inform the customer that the menu is currently unavailable."
    
    items = menu.get("items", [])
    categories = menu.get("categories", [])
    
    # Build formatted output
    lines = []
    lines.append("=== RESTAURANT MENU ===")
    lines.append("")
    
    # Group items by category
    if categories:
        for category in categories:
            # Category header
            lines.append(f"## {category.upper()}")
            lines.append("")
            
            # Items in this category
            category_items = [
                item for item in items
                if item.get("category") == category
            ]
            
            for item in category_items:
                lines.append(_format_menu_item(item))
            
            lines.append("")  # Spacing between categories
    else:
        # No categories - just list all items
        lines.append("## MENU ITEMS")
        lines.append("")
        for item in items:
            lines.append(_format_menu_item(item))
    
    lines.append("=== END MENU ===")
    
    return "\n".join(lines)


def _format_menu_item(item: Dict[str, Any]) -> str:
    """
    Format a single menu item for AI consumption.
    
    Args:
        item: Menu item dictionary
        
    Returns:
        Formatted item string
    """
    # Extract item details
    item_id = item.get("id", "unknown")
    name = item.get("name", "Unnamed Item")
    price = item.get("price", 0.0)
    description = item.get("description", "")
    
    # Build item line
    line = f"- {name} (${price:.2f}) [ID: {item_id}]"
    
    # Add description if available
    if description and description.strip():
        line += f"\n  Description: {description.strip()}"
    
    return line


# ============================================================================
# MENU FORMATTING VARIATIONS
# ============================================================================

def format_menu_compact(menu: Dict[str, Any]) -> str:
    """
    Format menu in compact form (minimal tokens).
    Useful when context window is tight.
    
    Args:
        menu: Menu dictionary from get_menu()
        
    Returns:
        Compact menu string
        
    Example:
        >>> compact = format_menu_compact(menu)
    """
    if not menu.get("has_items", False):
        return "Menu unavailable."
    
    items = menu.get("items", [])
    
    lines = ["MENU:"]
    for item in items:
        name = item.get("name", "Unknown")
        price = item.get("price", 0.0)
        item_id = item.get("id", "?")
        lines.append(f"{name} ${price:.2f} [{item_id}]")
    
    return "\n".join(lines)


def format_menu_by_category(menu: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Format menu as dictionary grouped by category.
    Useful for programmatic access in AI logic.
    
    Args:
        menu: Menu dictionary from get_menu()
        
    Returns:
        Dictionary mapping category -> list of items
        
    Example:
        >>> by_category = format_menu_by_category(menu)
        >>> pizzas = by_category.get("Pizza", [])
    """
    if not menu.get("has_items", False):
        return {}
    
    items = menu.get("items", [])
    categorized = {}
    
    for item in items:
        category = item.get("category", "Other")
        if category not in categorized:
            categorized[category] = []
        categorized[category].append(item)
    
    return categorized


# ============================================================================
# MENU SEARCH & FILTERING
# ============================================================================

def search_menu_items(
    menu: Dict[str, Any],
    query: str,
    search_fields: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Search menu items by name, description, or category.
    
    Args:
        menu: Menu dictionary from get_menu()
        query: Search term
        search_fields: Optional list of fields to search (default: name, description, category)
        
    Returns:
        List of matching menu items
        
    Example:
        >>> results = search_menu_items(menu, "pizza")
        >>> print(f"Found {len(results)} items")
    """
    if not menu.get("has_items", False):
        return []
    
    # Default search fields
    if search_fields is None:
        search_fields = ["name", "description", "category"]
    
    items = menu.get("items", [])
    query_lower = query.lower().strip()
    matches = []
    
    for item in items:
        # Check each search field
        for field in search_fields:
            value = item.get(field, "")
            if isinstance(value, str) and query_lower in value.lower():
                matches.append(item)
                break  # Don't add duplicates
    
    return matches


def filter_menu_by_category(
    menu: Dict[str, Any],
    category: str
) -> List[Dict[str, Any]]:
    """
    Filter menu items by category.
    
    Args:
        menu: Menu dictionary from get_menu()
        category: Category name to filter by
        
    Returns:
        List of items in the category
        
    Example:
        >>> pizzas = filter_menu_by_category(menu, "Pizza")
    """
    if not menu.get("has_items", False):
        return []
    
    items = menu.get("items", [])
    return [
        item for item in items
        if item.get("category", "").lower() == category.lower()
    ]


def filter_menu_by_price_range(
    menu: Dict[str, Any],
    min_price: Optional[float] = None,
    max_price: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Filter menu items by price range.
    
    Args:
        menu: Menu dictionary from get_menu()
        min_price: Minimum price (inclusive)
        max_price: Maximum price (inclusive)
        
    Returns:
        List of items within price range
        
    Example:
        >>> affordable = filter_menu_by_price_range(menu, max_price=10.00)
    """
    if not menu.get("has_items", False):
        return []
    
    items = menu.get("items", [])
    filtered = []
    
    for item in items:
        price = item.get("price", 0.0)
        
        # Check min price
        if min_price is not None and price < min_price:
            continue
        
        # Check max price
        if max_price is not None and price > max_price:
            continue
        
        filtered.append(item)
    
    return filtered


# ============================================================================
# MENU ITEM LOOKUP
# ============================================================================

def get_menu_item_by_id(
    menu: Dict[str, Any],
    item_id: str
) -> Optional[Dict[str, Any]]:
    """
    Get a specific menu item by ID.
    
    Args:
        menu: Menu dictionary from get_menu()
        item_id: Item identifier
        
    Returns:
        Menu item dictionary or None if not found
        
    Example:
        >>> item = get_menu_item_by_id(menu, "pizza_1")
        >>> if item:
        >>>     print(f"Found: {item['name']}")
    """
    if not menu.get("has_items", False):
        return None
    
    items = menu.get("items", [])
    
    for item in items:
        if item.get("id") == item_id:
            return item
    
    return None


def get_menu_item_by_name(
    menu: Dict[str, Any],
    name: str,
    fuzzy: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Get a menu item by name.
    
    Args:
        menu: Menu dictionary from get_menu()
        name: Item name to search for
        fuzzy: If True, use partial matching
        
    Returns:
        Menu item dictionary or None if not found
        
    Example:
        >>> item = get_menu_item_by_name(menu, "Margherita Pizza")
    """
    if not menu.get("has_items", False):
        return None
    
    items = menu.get("items", [])
    name_lower = name.lower().strip()
    
    for item in items:
        item_name = item.get("name", "").lower()
        
        if fuzzy:
            # Partial match
            if name_lower in item_name or item_name in name_lower:
                return item
        else:
            # Exact match
            if item_name == name_lower:
                return item
    
    return None


# ============================================================================
# MENU STATISTICS
# ============================================================================

def get_menu_stats(menu: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get statistics about the menu.
    
    Args:
        menu: Menu dictionary from get_menu()
        
    Returns:
        Dictionary with menu statistics
        
    Example:
        >>> stats = get_menu_stats(menu)
        >>> print(f"Average price: ${stats['avg_price']:.2f}")
    """
    if not menu.get("has_items", False):
        return {
            "total_items": 0,
            "total_categories": 0,
            "avg_price": 0.0,
            "min_price": 0.0,
            "max_price": 0.0,
            "price_range": 0.0
        }
    
    items = menu.get("items", [])
    prices = [item.get("price", 0.0) for item in items]
    
    return {
        "total_items": len(items),
        "total_categories": len(menu.get("categories", [])),
        "avg_price": sum(prices) / len(prices) if prices else 0.0,
        "min_price": min(prices) if prices else 0.0,
        "max_price": max(prices) if prices else 0.0,
        "price_range": max(prices) - min(prices) if prices else 0.0
    }


# ============================================================================
# MENU VALIDATION
# ============================================================================

def validate_menu_items(menu: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate menu items and return validation report.
    
    Args:
        menu: Menu dictionary from get_menu()
        
    Returns:
        Validation report with issues found
        
    Example:
        >>> report = validate_menu_items(menu)
        >>> if report["has_issues"]:
        >>>     print(f"Issues: {report['issues']}")
    """
    issues = []
    items = menu.get("items", [])
    
    for idx, item in enumerate(items):
        # Check required fields
        if not item.get("id"):
            issues.append(f"Item {idx}: Missing ID")
        
        if not item.get("name"):
            issues.append(f"Item {idx}: Missing name")
        
        price = item.get("price")
        if price is None:
            issues.append(f"Item {idx}: Missing price")
        elif price < 0:
            issues.append(f"Item {idx}: Negative price (${price})")
    
    return {
        "has_issues": len(issues) > 0,
        "issue_count": len(issues),
        "issues": issues,
        "valid_items": len(items) - len(issues)
    }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the menu module.
    """
    
    print("Menu Access Module Example")
    print("=" * 50)
    
    # Mock restaurant ID
    restaurant_id = "rest_123"
    
    try:
        # Fetch menu
        print(f"\nFetching menu for restaurant: {restaurant_id}")
        menu = get_menu(restaurant_id)
        
        print(f"✓ Menu loaded")
        print(f"  - Items: {menu['item_count']}")
        print(f"  - Categories: {len(menu['categories'])}")
        print(f"  - Has items: {menu['has_items']}")
        
        if menu['has_items']:
            print(f"\nCategories: {', '.join(menu['categories'])}")
            
            # Format for AI
            print("\n" + "=" * 50)
            print("Formatted for AI Prompt:")
            print("=" * 50)
            formatted = format_menu_for_prompt(menu)
            print(formatted)
            
            # Compact format
            print("\n" + "=" * 50)
            print("Compact Format:")
            print("=" * 50)
            compact = format_menu_compact(menu)
            print(compact)
            
            # Statistics
            print("\n" + "=" * 50)
            print("Menu Statistics:")
            print("=" * 50)
            stats = get_menu_stats(menu)
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: ${value:.2f}" if "price" in key else f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
            
            # Search example
            print("\n" + "=" * 50)
            print("Search Example:")
            print("=" * 50)
            results = search_menu_items(menu, "pizza")
            print(f"Search 'pizza': {len(results)} results")
            
            # Get item by ID (example)
            if menu['items']:
                first_item_id = menu['items'][0].get('id')
                item = get_menu_item_by_id(menu, first_item_id)
                if item:
                    print(f"\nItem lookup by ID '{first_item_id}':")
                    print(f"  Name: {item.get('name')}")
                    print(f"  Price: ${item.get('price', 0):.2f}")
            
            # Validation
            print("\n" + "=" * 50)
            print("Validation Report:")
            print("=" * 50)
            validation = validate_menu_items(menu)
            if validation['has_issues']:
                print(f"⚠ Found {validation['issue_count']} issues:")
                for issue in validation['issues']:
                    print(f"  - {issue}")
            else:
                print("✓ All items valid")
        
        else:
            print("\n⚠ Menu is empty")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
