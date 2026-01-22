"""
Menu Module (Production Hardened)
==================================
Split menu handling into data access and reasoning logic.

HARDENING UPDATES (v3.0):
✅ MenuRepository (data access, validation)
✅ MenuReasoner (interpretation, matching, normalization)
✅ Canonical item IDs (immutable identifiers)
✅ Synonym mapping for matching
✅ Hallucination rejection (AI cannot invent items)
✅ Deterministic validation
✅ Strict separation of data and logic

Version: 3.0.0 (Production Hardened)
Last Updated: 2026-01-22
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import defaultdict
import hashlib

try:
    from prometheus_client import Counter, Histogram, Gauge
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Cache configuration
CACHE_TTL = 300  # 5 minutes
CACHE_MAX_SIZE = 100

# Validation limits
MAX_MENU_SIZE = 1000
MAX_ITEM_NAME_LENGTH = 200
MAX_DESCRIPTION_LENGTH = 500
MIN_ITEM_PRICE = 0.01
MAX_ITEM_PRICE = 10000.00

# Matching configuration
MATCH_CONFIDENCE_THRESHOLD = 0.7  # 70% similarity
MAX_SYNONYMS_PER_ITEM = 10


# ============================================================================
# METRICS
# ============================================================================

if METRICS_ENABLED:
    menu_cache_hits = Counter('menu_cache_hits_total', 'Menu cache hits')
    menu_cache_misses = Counter('menu_cache_misses_total', 'Menu cache misses')
    menu_validation_errors = Counter(
        'menu_validation_errors_total',
        'Menu validation errors',
        ['error_type']
    )
    menu_hallucination_rejections = Counter(
        'menu_hallucination_rejections_total',
        'AI hallucination rejections'
    )
    menu_synonym_matches = Counter(
        'menu_synonym_matches_total',
        'Synonym-based matches'
    )
    menu_exact_matches = Counter(
        'menu_exact_matches_total',
        'Exact matches'
    )


# ============================================================================
# CANONICAL MENU ITEM
# ============================================================================

@dataclass(frozen=True)
class CanonicalMenuItem:
    """
    Canonical menu item with immutable identifier.
    
    CRITICAL: frozen=True means item CANNOT be modified.
    This prevents AI from altering menu data.
    """
    canonical_id: str  # Immutable, unique identifier
    name: str
    price: float
    description: str
    category: str
    available: bool
    synonyms: Tuple[str, ...] = field(default_factory=tuple)  # Immutable tuple
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "canonical_id": self.canonical_id,
            "name": self.name,
            "price": self.price,
            "description": self.description,
            "category": self.category,
            "available": self.available,
            "synonyms": list(self.synonyms),
            "metadata": self.metadata.copy()
        }
    
    def matches_query(self, query: str) -> bool:
        """
        Check if query matches this item.
        
        Matches against: name, synonyms
        """
        query_lower = query.lower().strip()
        
        # Exact name match
        if query_lower == self.name.lower():
            return True
        
        # Synonym match
        for synonym in self.synonyms:
            if query_lower == synonym.lower():
                return True
        
        # Partial match (contains)
        if query_lower in self.name.lower():
            return True
        
        return False


# ============================================================================
# MENU REPOSITORY (Data Access & Validation)
# ============================================================================

class MenuRepository:
    """
    Data access layer for menu management.
    
    Responsibilities:
    - Load menu data
    - Validate menu structure
    - Cache validated menus
    - Generate canonical IDs
    - Store synonym mappings
    
    Does NOT:
    - Interpret user queries
    - Match items
    - Reason about menu
    """
    
    def __init__(self, ttl: int = CACHE_TTL, max_size: int = CACHE_MAX_SIZE):
        self.cache: Dict[str, Tuple[Dict[str, Any], datetime]] = {}
        self.ttl = ttl
        self.max_size = max_size
        
        # Canonical item registry (by restaurant)
        self._canonical_items: Dict[str, Dict[str, CanonicalMenuItem]] = {}
        
        # Synonym index (for fast lookup)
        self._synonym_index: Dict[str, Dict[str, Set[str]]] = {}  # restaurant_id -> {synonym: [canonical_ids]}
    
    async def get_menu(self, restaurant_id: str) -> Dict[str, Any]:
        """
        Get validated menu for restaurant.
        
        Args:
            restaurant_id: Restaurant identifier
            
        Returns:
            Validated menu with canonical items
        """
        # Check cache
        cached = self._get_from_cache(restaurant_id)
        if cached:
            return cached
        
        try:
            # Fetch raw menu
            raw_menu = await self._fetch_menu_from_db(restaurant_id)
            
            # Validate and canonicalize
            validated = self._validate_and_canonicalize(raw_menu, restaurant_id)
            
            # Cache
            self._set_in_cache(restaurant_id, validated)
            
            return validated
        
        except Exception as e:
            logger.error(f"Error loading menu for {restaurant_id}: {str(e)}")
            return self._get_fallback_menu(restaurant_id)
    
    def get_canonical_item(
        self,
        restaurant_id: str,
        canonical_id: str
    ) -> Optional[CanonicalMenuItem]:
        """
        Get canonical item by ID.
        
        Args:
            restaurant_id: Restaurant identifier
            canonical_id: Canonical item ID
            
        Returns:
            CanonicalMenuItem or None
        """
        items = self._canonical_items.get(restaurant_id, {})
        return items.get(canonical_id)
    
    def get_all_canonical_items(
        self,
        restaurant_id: str
    ) -> List[CanonicalMenuItem]:
        """
        Get all canonical items for restaurant.
        
        Args:
            restaurant_id: Restaurant identifier
            
        Returns:
            List of CanonicalMenuItem
        """
        items = self._canonical_items.get(restaurant_id, {})
        return list(items.values())
    
    def invalidate_cache(self, restaurant_id: str):
        """Invalidate cache for restaurant."""
        if restaurant_id in self.cache:
            del self.cache[restaurant_id]
            logger.info(f"Menu cache invalidated: {restaurant_id}")
    
    def _get_from_cache(self, restaurant_id: str) -> Optional[Dict[str, Any]]:
        """Get menu from cache if valid."""
        if restaurant_id not in self.cache:
            if METRICS_ENABLED:
                menu_cache_misses.inc()
            return None
        
        menu, timestamp = self.cache[restaurant_id]
        
        # Check TTL
        if datetime.utcnow() - timestamp > timedelta(seconds=self.ttl):
            del self.cache[restaurant_id]
            if METRICS_ENABLED:
                menu_cache_misses.inc()
            return None
        
        if METRICS_ENABLED:
            menu_cache_hits.inc()
        
        return menu
    
    def _set_in_cache(self, restaurant_id: str, menu: Dict[str, Any]):
        """Set menu in cache."""
        # Evict oldest if full
        if len(self.cache) >= self.max_size:
            oldest = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest]
        
        self.cache[restaurant_id] = (menu, datetime.utcnow())
    
    async def _fetch_menu_from_db(self, restaurant_id: str) -> Dict[str, Any]:
        """Fetch raw menu from database."""
        try:
            from db import db
            
            loop = asyncio.get_event_loop()
            menu = await loop.run_in_executor(
                None,
                lambda: db.fetch_menu(restaurant_id)
            )
            
            if not menu:
                logger.warning(f"No menu found: {restaurant_id}")
                return self._get_fallback_menu(restaurant_id)
            
            return menu
        
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            return self._get_fallback_menu(restaurant_id)
    
    def _validate_and_canonicalize(
        self,
        raw_menu: Dict[str, Any],
        restaurant_id: str
    ) -> Dict[str, Any]:
        """
        Validate menu and generate canonical items.
        
        This is DETERMINISTIC - same input always produces same output.
        """
        if not raw_menu or not isinstance(raw_menu, dict):
            return {"items": [], "categories": []}
        
        # Validate items
        raw_items = raw_menu.get("items", [])
        if not isinstance(raw_items, list):
            raw_items = []
        
        canonical_items = []
        canonical_registry = {}
        synonym_index = defaultdict(set)
        
        for raw_item in raw_items[:MAX_MENU_SIZE]:
            try:
                # Generate canonical ID (deterministic)
                canonical_id = self._generate_canonical_id(
                    restaurant_id,
                    raw_item.get("id", ""),
                    raw_item.get("name", "")
                )
                
                # Validate item
                validated = self._validate_item(raw_item)
                if not validated:
                    continue
                
                # Extract synonyms
                synonyms = self._extract_synonyms(validated)
                
                # Create canonical item (immutable)
                canonical = CanonicalMenuItem(
                    canonical_id=canonical_id,
                    name=validated["name"],
                    price=validated["price"],
                    description=validated["description"],
                    category=validated["category"],
                    available=validated["available"],
                    synonyms=tuple(synonyms),
                    metadata=validated.get("metadata", {})
                )
                
                canonical_items.append(canonical)
                canonical_registry[canonical_id] = canonical
                
                # Index synonyms
                for synonym in synonyms:
                    synonym_index[synonym.lower()].add(canonical_id)
            
            except Exception as e:
                logger.error(f"Error canonicalizing item: {str(e)}")
                if METRICS_ENABLED:
                    menu_validation_errors.labels(error_type='canonicalization').inc()
        
        # Store canonical items
        self._canonical_items[restaurant_id] = canonical_registry
        self._synonym_index[restaurant_id] = dict(synonym_index)
        
        # Extract categories
        categories = list(set(item.category for item in canonical_items))
        
        return {
            "items": [item.to_dict() for item in canonical_items],
            "categories": sorted(categories),
            "restaurant_id": restaurant_id,
            "canonical_count": len(canonical_items),
            "validated_at": datetime.utcnow().isoformat()
        }
    
    def _generate_canonical_id(
        self,
        restaurant_id: str,
        item_id: str,
        item_name: str
    ) -> str:
        """
        Generate deterministic canonical ID.
        
        Same inputs always produce same ID.
        """
        # Use hash of restaurant + item_id + name
        content = f"{restaurant_id}:{item_id}:{item_name}"
        hash_digest = hashlib.sha256(content.encode()).hexdigest()
        return f"item_{hash_digest[:16]}"
    
    def _validate_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validate menu item (deterministic).
        
        Returns validated item or None.
        """
        if not isinstance(item, dict):
            return None
        
        # Required fields
        if "name" not in item or "price" not in item:
            if METRICS_ENABLED:
                menu_validation_errors.labels(error_type='missing_fields').inc()
            return None
        
        # Validate name
        name = str(item["name"]).strip()
        if not name or len(name) > MAX_ITEM_NAME_LENGTH:
            if METRICS_ENABLED:
                menu_validation_errors.labels(error_type='invalid_name').inc()
            return None
        
        # Validate price
        price = self._normalize_price(item["price"])
        if price is None:
            if METRICS_ENABLED:
                menu_validation_errors.labels(error_type='invalid_price').inc()
            return None
        
        if price < MIN_ITEM_PRICE or price > MAX_ITEM_PRICE:
            if METRICS_ENABLED:
                menu_validation_errors.labels(error_type='price_out_of_range').inc()
            return None
        
        # Optional fields
        description = str(item.get("description", "")).strip()
        if len(description) > MAX_DESCRIPTION_LENGTH:
            description = description[:MAX_DESCRIPTION_LENGTH] + "..."
        
        category = str(item.get("category", "Other")).strip()
        available = bool(item.get("available", True))
        
        return {
            "name": name,
            "price": price,
            "description": description,
            "category": category,
            "available": available,
            "metadata": item.get("metadata", {})
        }
    
    def _normalize_price(self, price: Any) -> Optional[float]:
        """Normalize price to float (deterministic)."""
        try:
            if isinstance(price, (int, float)):
                price_float = float(price)
            elif isinstance(price, str):
                price_clean = price.replace("$", "").replace(",", "").strip()
                price_float = float(price_clean)
            else:
                return None
            
            if price_float < 0 or price_float > 10000:
                return None
            
            return round(price_float, 2)
        
        except (ValueError, TypeError):
            return None
    
    def _extract_synonyms(self, item: Dict[str, Any]) -> List[str]:
        """
        Extract synonyms from item metadata.
        
        Synonyms are explicitly defined, NOT generated by AI.
        """
        synonyms = []
        
        # Check metadata for explicit synonyms
        metadata = item.get("metadata", {})
        if "synonyms" in metadata and isinstance(metadata["synonyms"], list):
            for syn in metadata["synonyms"][:MAX_SYNONYMS_PER_ITEM]:
                if isinstance(syn, str) and syn.strip():
                    synonyms.append(syn.strip())
        
        return synonyms
    
    def _get_fallback_menu(self, restaurant_id: str) -> Dict[str, Any]:
        """Get safe fallback menu."""
        return {
            "items": [],
            "categories": [],
            "restaurant_id": restaurant_id,
            "fallback": True,
            "validated_at": datetime.utcnow().isoformat()
        }


# ============================================================================
# MENU REASONER (Interpretation & Matching)
# ============================================================================

class MenuReasoner:
    """
    Logic layer for menu interpretation and matching.
    
    Responsibilities:
    - Interpret user queries
    - Match queries to canonical items
    - Normalize item references
    - Reject hallucinations
    
    Does NOT:
    - Access database
    - Validate data
    - Modify menu items
    """
    
    def __init__(self, repository: MenuRepository):
        self.repository = repository
    
    def match_item(
        self,
        restaurant_id: str,
        query: str,
        category: Optional[str] = None
    ) -> Optional[CanonicalMenuItem]:
        """
        Match user query to canonical menu item.
        
        CRITICAL: Returns None if no match (hallucination rejection).
        
        Args:
            restaurant_id: Restaurant identifier
            query: User query (e.g., "large pizza")
            category: Optional category filter
            
        Returns:
            CanonicalMenuItem or None
        """
        if not query or not query.strip():
            return None
        
        query_clean = query.strip()
        
        # Get all canonical items
        items = self.repository.get_all_canonical_items(restaurant_id)
        if not items:
            return None
        
        # Filter by category if specified
        if category:
            items = [item for item in items if item.category == category]
        
        # 1. Try exact match (highest priority)
        for item in items:
            if item.name.lower() == query_clean.lower():
                if METRICS_ENABLED:
                    menu_exact_matches.inc()
                return item
        
        # 2. Try synonym match
        synonym_match = self._match_by_synonym(restaurant_id, query_clean, items)
        if synonym_match:
            if METRICS_ENABLED:
                menu_synonym_matches.inc()
            return synonym_match
        
        # 3. Try partial match (contains)
        for item in items:
            if query_clean.lower() in item.name.lower():
                return item
        
        # 4. No match - hallucination rejection
        logger.warning(
            f"No menu match for query: '{query}' (restaurant: {restaurant_id})"
        )
        
        if METRICS_ENABLED:
            menu_hallucination_rejections.inc()
        
        return None  # AI cannot invent items!
    
    def match_items(
        self,
        restaurant_id: str,
        queries: List[str]
    ) -> List[Optional[CanonicalMenuItem]]:
        """
        Match multiple queries (batch).
        
        Returns list with same length as queries.
        None for items that don't match (hallucination rejection).
        """
        return [self.match_item(restaurant_id, query) for query in queries]
    
    def validate_item_exists(
        self,
        restaurant_id: str,
        canonical_id: str
    ) -> bool:
        """
        Validate that item exists (hallucination check).
        
        Use this to verify AI-suggested items.
        
        Args:
            restaurant_id: Restaurant identifier
            canonical_id: Canonical item ID
            
        Returns:
            True if item exists
        """
        item = self.repository.get_canonical_item(restaurant_id, canonical_id)
        return item is not None
    
    def get_available_items(
        self,
        restaurant_id: str,
        category: Optional[str] = None
    ) -> List[CanonicalMenuItem]:
        """
        Get available items (filtering).
        
        Args:
            restaurant_id: Restaurant identifier
            category: Optional category filter
            
        Returns:
            List of available canonical items
        """
        items = self.repository.get_all_canonical_items(restaurant_id)
        
        # Filter available
        items = [item for item in items if item.available]
        
        # Filter category
        if category:
            items = [item for item in items if item.category == category]
        
        return items
    
    def get_categories(self, restaurant_id: str) -> List[str]:
        """
        Get all categories.
        
        Args:
            restaurant_id: Restaurant identifier
            
        Returns:
            List of category names
        """
        items = self.repository.get_all_canonical_items(restaurant_id)
        categories = set(item.category for item in items)
        return sorted(categories)
    
    def format_for_ai(
        self,
        restaurant_id: str,
        max_items: Optional[int] = None
    ) -> str:
        """
        Format menu for AI prompt.
        
        ONLY includes canonical items (no hallucinations possible).
        
        Args:
            restaurant_id: Restaurant identifier
            max_items: Max items to include
            
        Returns:
            Formatted menu string
        """
        items = self.get_available_items(restaurant_id)
        
        if not items:
            return "Menu unavailable."
        
        # Limit items
        if max_items and len(items) > max_items:
            items = items[:max_items]
        
        # Group by category
        by_category = defaultdict(list)
        for item in items:
            by_category[item.category].append(item)
        
        # Format
        lines = []
        for category in sorted(by_category.keys()):
            lines.append(f"\n{category}:")
            lines.append("-" * 40)
            
            for item in by_category[category]:
                # Include canonical ID for validation
                item_line = f"• {item.name} - ${item.price:.2f} [ID: {item.canonical_id}]"
                
                if item.description:
                    item_line += f"\n  {item.description}"
                
                lines.append(item_line)
        
        return "\n".join(lines)
    
    def _match_by_synonym(
        self,
        restaurant_id: str,
        query: str,
        items: List[CanonicalMenuItem]
    ) -> Optional[CanonicalMenuItem]:
        """Match by synonym."""
        query_lower = query.lower()
        
        # Check synonym index
        synonym_index = self.repository._synonym_index.get(restaurant_id, {})
        canonical_ids = synonym_index.get(query_lower, set())
        
        if not canonical_ids:
            return None
        
        # Get first matching canonical item
        for canonical_id in canonical_ids:
            item = self.repository.get_canonical_item(restaurant_id, canonical_id)
            if item and item.available:
                return item
        
        return None


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

_repository = MenuRepository()
_reasoner = MenuReasoner(_repository)


# ============================================================================
# PUBLIC API
# ============================================================================

async def get_menu(restaurant_id: str) -> Dict[str, Any]:
    """
    Get validated menu (data access).
    
    Args:
        restaurant_id: Restaurant identifier
        
    Returns:
        Validated menu with canonical items
    """
    return await _repository.get_menu(restaurant_id)


def match_menu_item(
    restaurant_id: str,
    query: str,
    category: Optional[str] = None
) -> Optional[CanonicalMenuItem]:
    """
    Match user query to menu item (reasoning).
    
    CRITICAL: Returns None if no match (hallucination rejection).
    
    Args:
        restaurant_id: Restaurant identifier
        query: User query
        category: Optional category filter
        
    Returns:
        CanonicalMenuItem or None
    """
    return _reasoner.match_item(restaurant_id, query, category)


def validate_item_exists(restaurant_id: str, canonical_id: str) -> bool:
    """
    Validate that item exists (hallucination check).
    
    Use this to verify AI-suggested items BEFORE accepting them.
    
    Args:
        restaurant_id: Restaurant identifier
        canonical_id: Canonical item ID
        
    Returns:
        True if item exists
    """
    return _reasoner.validate_item_exists(restaurant_id, canonical_id)


def get_available_items(
    restaurant_id: str,
    category: Optional[str] = None
) -> List[CanonicalMenuItem]:
    """
    Get available menu items (reasoning).
    
    Args:
        restaurant_id: Restaurant identifier
        category: Optional category filter
        
    Returns:
        List of available items
    """
    return _reasoner.get_available_items(restaurant_id, category)


def format_menu_for_ai(
    restaurant_id: str,
    max_items: Optional[int] = None
) -> str:
    """
    Format menu for AI prompt (reasoning).
    
    Args:
        restaurant_id: Restaurant identifier
        max_items: Max items to include
        
    Returns:
        Formatted menu string with canonical IDs
    """
    return _reasoner.format_for_ai(restaurant_id, max_items)


def invalidate_menu_cache(restaurant_id: str):
    """Invalidate menu cache (data access)."""
    _repository.invalidate_cache(restaurant_id)


def get_menu_categories(restaurant_id: str) -> List[str]:
    """Get menu categories (reasoning)."""
    return _reasoner.get_categories(restaurant_id)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def example():
        print("Menu Module (Production Hardened v3.0)")
        print("="*60)
        
        # Create sample menu
        sample_menu = {
            "restaurant_id": "rest_001",
            "items": [
                {
                    "id": "1",
                    "name": "Large Pizza",
                    "price": 15.99,
                    "description": "Classic large pizza",
                    "category": "Pizza",
                    "available": True,
                    "metadata": {
                        "synonyms": ["pizza", "large pie", "family pizza"]
                    }
                },
                {
                    "id": "2",
                    "name": "Caesar Salad",
                    "price": 8.99,
                    "description": "Fresh romaine lettuce",
                    "category": "Salads",
                    "available": True,
                    "metadata": {
                        "synonyms": ["salad", "caesar"]
                    }
                },
                {
                    "id": "3",
                    "name": "Chicken Wings",
                    "price": 12.99,
                    "description": "Spicy buffalo wings",
                    "category": "Appetizers",
                    "available": False
                }
            ]
        }
        
        # Load menu into repository
        validated = _repository._validate_and_canonicalize(sample_menu, "rest_001")
        _repository._set_in_cache("rest_001", validated)
        
        print("\n1. Canonical Items:")
        items = _repository.get_all_canonical_items("rest_001")
        for item in items:
            print(f"  • {item.name} [{item.canonical_id}]")
            print(f"    Price: ${item.price}")
            print(f"    Synonyms: {item.synonyms}")
            print(f"    Available: {item.available}")
        
        print("\n2. Exact Match:")
        match = match_menu_item("rest_001", "Large Pizza")
        if match:
            print(f"  ✓ Matched: {match.name} [{match.canonical_id}]")
        
        print("\n3. Synonym Match:")
        match = match_menu_item("rest_001", "pizza")
        if match:
            print(f"  ✓ Matched: {match.name} via synonym")
        
        print("\n4. Hallucination Rejection:")
        match = match_menu_item("rest_001", "Unicorn Burger")
        if match is None:
            print(f"  ✓ Correctly rejected: 'Unicorn Burger' (not on menu)")
        
        print("\n5. Validation Check:")
        # Get canonical ID
        items = _repository.get_all_canonical_items("rest_001")
        valid_id = items[0].canonical_id
        fake_id = "item_fake123"
        
        print(f"  Valid ID '{valid_id}': {validate_item_exists('rest_001', valid_id)}")
        print(f"  Fake ID '{fake_id}': {validate_item_exists('rest_001', fake_id)}")
        
        print("\n6. Available Items Only:")
        available = get_available_items("rest_001")
        print(f"  {len(available)} available items:")
        for item in available:
            print(f"    • {item.name}")
        
        print("\n7. AI-Formatted Menu:")
        formatted = format_menu_for_ai("rest_001")
        print(formatted)
    
    asyncio.run(example())
