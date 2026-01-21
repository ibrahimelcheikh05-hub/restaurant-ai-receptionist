"""
Menu Module (Production)
=========================
Hardened menu management with caching, validation, fallbacks.
Safe prompt formatting, price normalization, never crashes calls.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio


logger = logging.getLogger(__name__)


# Cache configuration
CACHE_TTL = 300  # 5 minutes
CACHE_MAX_SIZE = 100

# Validation limits
MAX_MENU_SIZE = 1000  # Max items per menu
MAX_ITEM_NAME_LENGTH = 200
MAX_DESCRIPTION_LENGTH = 500


class MenuCache:
    """Simple TTL-based menu cache."""
    
    def __init__(self, ttl: int = CACHE_TTL, max_size: int = CACHE_MAX_SIZE):
        self.cache: Dict[str, tuple[Dict[str, Any], datetime]] = {}
        self.ttl = ttl
        self.max_size = max_size
    
    def get(self, restaurant_id: str) -> Optional[Dict[str, Any]]:
        """Get cached menu if valid."""
        if restaurant_id not in self.cache:
            return None
        
        menu, timestamp = self.cache[restaurant_id]
        
        # Check TTL
        if datetime.utcnow() - timestamp > timedelta(seconds=self.ttl):
            del self.cache[restaurant_id]
            logger.debug(f"Menu cache expired for {restaurant_id}")
            return None
        
        logger.debug(f"Menu cache hit for {restaurant_id}")
        return menu
    
    def set(self, restaurant_id: str, menu: Dict[str, Any]):
        """Cache menu with TTL."""
        # Evict oldest if cache full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
            logger.debug(f"Evicted oldest cache entry: {oldest_key}")
        
        self.cache[restaurant_id] = (menu, datetime.utcnow())
        logger.debug(f"Menu cached for {restaurant_id}")
    
    def invalidate(self, restaurant_id: str):
        """Invalidate cache for restaurant."""
        if restaurant_id in self.cache:
            del self.cache[restaurant_id]
            logger.info(f"Menu cache invalidated for {restaurant_id}")
    
    def clear(self):
        """Clear entire cache."""
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"Menu cache cleared ({count} entries)")
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)


_menu_cache = MenuCache()


async def get_menu(restaurant_id: str) -> Dict[str, Any]:
    """
    Get menu for restaurant with caching and validation.
    
    Args:
        restaurant_id: Restaurant identifier
        
    Returns:
        Validated menu dictionary
        
    Example:
        >>> menu = await get_menu("rest_001")
        >>> print(menu["items"])
    """
    # Check cache first
    cached = _menu_cache.get(restaurant_id)
    if cached:
        return cached
    
    try:
        # Fetch menu from database
        menu = await _fetch_menu_from_db(restaurant_id)
        
        # Validate menu
        validated = _validate_menu(menu)
        
        # Cache validated menu
        _menu_cache.set(restaurant_id, validated)
        
        return validated
    
    except Exception as e:
        logger.error(f"Error fetching menu for {restaurant_id}: {str(e)}")
        
        # Return safe fallback
        return _get_fallback_menu(restaurant_id)


async def _fetch_menu_from_db(restaurant_id: str) -> Dict[str, Any]:
    """
    Fetch menu from database.
    
    Args:
        restaurant_id: Restaurant identifier
        
    Returns:
        Raw menu data
    """
    try:
        from db import db
        
        # Run blocking DB call in executor
        loop = asyncio.get_event_loop()
        menu = await loop.run_in_executor(
            None,
            lambda: db.fetch_menu(restaurant_id)
        )
        
        if not menu:
            logger.warning(f"No menu found for {restaurant_id}")
            return _get_fallback_menu(restaurant_id)
        
        logger.info(f"Fetched menu for {restaurant_id}: {len(menu.get('items', []))} items")
        return menu
    
    except Exception as e:
        logger.error(f"Database error fetching menu: {str(e)}")
        return _get_fallback_menu(restaurant_id)


def _validate_menu(menu: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize menu data.
    
    Args:
        menu: Raw menu data
        
    Returns:
        Validated menu data
    """
    if not menu or not isinstance(menu, dict):
        logger.warning("Invalid menu structure")
        return {"items": [], "categories": []}
    
    # Validate items
    items = menu.get("items", [])
    if not isinstance(items, list):
        logger.warning("Menu items is not a list")
        items = []
    
    validated_items = []
    
    for item in items[:MAX_MENU_SIZE]:  # Enforce size limit
        try:
            validated_item = _validate_menu_item(item)
            if validated_item:
                validated_items.append(validated_item)
        except Exception as e:
            logger.error(f"Error validating menu item: {str(e)}")
            continue
    
    # Validate categories
    categories = menu.get("categories", [])
    if not isinstance(categories, list):
        categories = []
    
    validated_categories = [
        cat for cat in categories
        if isinstance(cat, str) and len(cat) > 0
    ]
    
    return {
        "items": validated_items,
        "categories": validated_categories,
        "restaurant_id": menu.get("restaurant_id", "unknown"),
        "validated_at": datetime.utcnow().isoformat()
    }


def _validate_menu_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Validate individual menu item.
    
    Args:
        item: Menu item data
        
    Returns:
        Validated item or None if invalid
    """
    if not isinstance(item, dict):
        return None
    
    # Required fields
    if "name" not in item or "price" not in item:
        logger.warning("Menu item missing required fields")
        return None
    
    name = str(item["name"]).strip()
    if not name or len(name) > MAX_ITEM_NAME_LENGTH:
        logger.warning(f"Invalid item name: {name}")
        return None
    
    # Normalize price
    price = _normalize_price(item["price"])
    if price is None:
        logger.warning(f"Invalid price for item {name}")
        return None
    
    # Optional fields
    description = str(item.get("description", "")).strip()
    if len(description) > MAX_DESCRIPTION_LENGTH:
        description = description[:MAX_DESCRIPTION_LENGTH] + "..."
    
    category = str(item.get("category", "Other")).strip()
    available = bool(item.get("available", True))
    
    return {
        "id": str(item.get("id", "")),
        "name": name,
        "price": price,
        "description": description,
        "category": category,
        "available": available,
        "metadata": item.get("metadata", {})
    }


def _normalize_price(price: Any) -> Optional[float]:
    """
    Normalize price to float.
    
    Args:
        price: Raw price value
        
    Returns:
        Normalized price or None if invalid
    """
    try:
        # Handle various price formats
        if isinstance(price, (int, float)):
            price_float = float(price)
        elif isinstance(price, str):
            # Remove currency symbols and commas
            price_clean = price.replace("$", "").replace(",", "").strip()
            price_float = float(price_clean)
        else:
            return None
        
        # Validate range
        if price_float < 0 or price_float > 10000:
            logger.warning(f"Price out of range: {price_float}")
            return None
        
        # Round to 2 decimal places
        return round(price_float, 2)
    
    except (ValueError, TypeError) as e:
        logger.error(f"Price normalization error: {str(e)}")
        return None


def _get_fallback_menu(restaurant_id: str) -> Dict[str, Any]:
    """
    Get safe fallback menu.
    
    Args:
        restaurant_id: Restaurant identifier
        
    Returns:
        Minimal fallback menu
    """
    logger.warning(f"Using fallback menu for {restaurant_id}")
    
    return {
        "items": [
            {
                "id": "fallback_1",
                "name": "Special of the Day",
                "price": 0.00,
                "description": "Please ask for today's specials",
                "category": "Specials",
                "available": True,
                "metadata": {"fallback": True}
            }
        ],
        "categories": ["Specials"],
        "restaurant_id": restaurant_id,
        "fallback": True,
        "validated_at": datetime.utcnow().isoformat()
    }


def format_menu_for_prompt(menu: Dict[str, Any], max_items: Optional[int] = None) -> str:
    """
    Format menu for LLM prompt.
    
    Args:
        menu: Menu data
        max_items: Maximum items to include (optional)
        
    Returns:
        Formatted menu string
        
    Example:
        >>> menu_text = format_menu_for_prompt(menu, max_items=10)
    """
    if not menu or not menu.get("items"):
        return "Menu unavailable. Please ask customer what they'd like."
    
    items = menu["items"]
    
    # Limit items if requested
    if max_items and len(items) > max_items:
        items = items[:max_items]
    
    # Group by category
    categorized = {}
    for item in items:
        if not item.get("available", True):
            continue
        
        category = item.get("category", "Other")
        if category not in categorized:
            categorized[category] = []
        categorized[category].append(item)
    
    # Format output
    lines = []
    
    for category, items_in_cat in sorted(categorized.items()):
        lines.append(f"\n{category}:")
        lines.append("-" * 40)
        
        for item in items_in_cat:
            name = item["name"]
            price = item["price"]
            description = item.get("description", "")
            
            # Format item line
            item_line = f"• {name} - ${price:.2f}"
            
            if description:
                item_line += f"\n  {description}"
            
            lines.append(item_line)
    
    return "\n".join(lines)


def format_menu_compact(menu: Dict[str, Any]) -> str:
    """
    Format menu in compact format (names and prices only).
    
    Args:
        menu: Menu data
        
    Returns:
        Compact menu string
    """
    if not menu or not menu.get("items"):
        return "Menu unavailable"
    
    items = [
        f"{item['name']} (${item['price']:.2f})"
        for item in menu["items"]
        if item.get("available", True)
    ]
    
    return ", ".join(items)


def search_menu_items(
    menu: Dict[str, Any],
    query: str,
    category: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search menu items by name or description.
    
    Args:
        menu: Menu data
        query: Search query
        category: Filter by category (optional)
        
    Returns:
        Matching items
    """
    if not menu or not menu.get("items"):
        return []
    
    query_lower = query.lower()
    results = []
    
    for item in menu["items"]:
        if not item.get("available", True):
            continue
        
        # Category filter
        if category and item.get("category") != category:
            continue
        
        # Search in name and description
        name = item.get("name", "").lower()
        description = item.get("description", "").lower()
        
        if query_lower in name or query_lower in description:
            results.append(item)
    
    return results


def get_item_by_id(menu: Dict[str, Any], item_id: str) -> Optional[Dict[str, Any]]:
    """
    Get menu item by ID.
    
    Args:
        menu: Menu data
        item_id: Item identifier
        
    Returns:
        Menu item or None
    """
    if not menu or not menu.get("items"):
        return None
    
    for item in menu["items"]:
        if item.get("id") == item_id:
            return item
    
    return None


def get_item_by_name(menu: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    """
    Get menu item by name (case-insensitive).
    
    Args:
        menu: Menu data
        name: Item name
        
    Returns:
        Menu item or None
    """
    if not menu or not menu.get("items"):
        return None
    
    name_lower = name.lower()
    
    for item in menu["items"]:
        if item.get("name", "").lower() == name_lower:
            return item
    
    return None


def get_items_by_category(menu: Dict[str, Any], category: str) -> List[Dict[str, Any]]:
    """
    Get all items in a category.
    
    Args:
        menu: Menu data
        category: Category name
        
    Returns:
        List of items
    """
    if not menu or not menu.get("items"):
        return []
    
    return [
        item for item in menu["items"]
        if item.get("category") == category and item.get("available", True)
    ]


def get_categories(menu: Dict[str, Any]) -> List[str]:
    """
    Get list of categories.
    
    Args:
        menu: Menu data
        
    Returns:
        List of category names
    """
    if not menu:
        return []
    
    # Use explicit categories if available
    if menu.get("categories"):
        return menu["categories"]
    
    # Otherwise extract from items
    categories = set()
    for item in menu.get("items", []):
        cat = item.get("category")
        if cat:
            categories.add(cat)
    
    return sorted(list(categories))


def invalidate_menu_cache(restaurant_id: str):
    """
    Invalidate cached menu.
    
    Args:
        restaurant_id: Restaurant identifier
    """
    _menu_cache.invalidate(restaurant_id)


def clear_menu_cache():
    """Clear all cached menus."""
    _menu_cache.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get menu cache statistics."""
    return {
        "size": _menu_cache.size(),
        "max_size": _menu_cache.max_size,
        "ttl_seconds": _menu_cache.ttl
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def example():
        print("Menu Module (Production)")
        print("=" * 50)
        
        # Create sample menu
        sample_menu = {
            "restaurant_id": "rest_001",
            "items": [
                {
                    "id": "1",
                    "name": "Large Pizza",
                    "price": 15.99,
                    "description": "Classic large pizza with your choice of toppings",
                    "category": "Pizza",
                    "available": True
                },
                {
                    "id": "2",
                    "name": "Chicken Wings",
                    "price": "$9.99",  # Test price normalization
                    "description": "Crispy chicken wings",
                    "category": "Appetizers",
                    "available": True
                },
                {
                    "id": "3",
                    "name": "Caesar Salad",
                    "price": 8.50,
                    "description": "Fresh romaine lettuce with caesar dressing",
                    "category": "Salads",
                    "available": False  # Unavailable
                }
            ],
            "categories": ["Pizza", "Appetizers", "Salads"]
        }
        
        # Validate
        validated = _validate_menu(sample_menu)
        print(f"\nValidated menu: {len(validated['items'])} items")
        
        # Format for prompt
        print("\nFormatted for LLM:")
        print(format_menu_for_prompt(validated))
        
        # Compact format
        print("\nCompact format:")
        print(format_menu_compact(validated))
        
        # Search
        print("\nSearch 'pizza':")
        results = search_menu_items(validated, "pizza")
        for item in results:
            print(f"  - {item['name']} (${item['price']:.2f})")
        
        # Categories
        print("\nCategories:", get_categories(validated))
        
        # Cache stats
        print("\nCache stats:", get_cache_stats())
        
        print("\n" + "=" * 50)
        print("Production menu module ready")
    
    asyncio.run(example())
