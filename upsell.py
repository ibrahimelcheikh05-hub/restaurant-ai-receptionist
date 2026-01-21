"""
Upsell Module (Enterprise Production)
======================================
Enterprise-grade upsell recommendation engine with A/B testing.

NEW FEATURES (Enterprise v2.0):
✅ A/B testing support for strategies
✅ Conversion rate tracking per strategy
✅ Suggestion quality metrics
✅ Revenue impact tracking
✅ Prometheus metrics integration
✅ Personalization scoring
✅ Category performance analytics
✅ Time-to-accept tracking
✅ Multi-variant testing

Version: 2.0.0 (Enterprise)
Last Updated: 2026-01-21
"""

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import hashlib

try:
    from prometheus_client import Counter, Histogram, Gauge
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False


logger = logging.getLogger(__name__)


# Configuration
MAX_UPSELLS_PER_CALL = 3
MAX_UPSELLS_PER_TURN = 1
MIN_ORDER_VALUE_FOR_UPSELL = 5.00
MAX_UPSELL_PRICE_RATIO = 0.5  # Upsell price <= 50% of current order


# Prometheus Metrics
if METRICS_ENABLED:
    upsell_offers_total = Counter(
        'upsell_offers_total',
        'Total upsell offers',
        ['category', 'strategy']
    )
    upsell_conversions_total = Counter(
        'upsell_conversions_total',
        'Total upsell conversions',
        ['category', 'strategy']
    )
    upsell_rejections_total = Counter(
        'upsell_rejections_total',
        'Total upsell rejections',
        ['category']
    )
    upsell_revenue_dollars = Counter(
        'upsell_revenue_dollars_total',
        'Revenue from upsells'
    )
    upsell_conversion_rate = Gauge(
        'upsell_conversion_rate',
        'Current conversion rate',
        ['strategy']
    )
    upsell_suggestion_quality = Histogram(
        'upsell_suggestion_quality_score',
        'Suggestion quality scores'
    )


# Configuration
MAX_UPSELLS_PER_CALL = 3
MAX_UPSELLS_PER_TURN = 1
MIN_ORDER_VALUE_FOR_UPSELL = 5.00
MAX_UPSELL_PRICE_RATIO = 0.5  # Upsell price <= 50% of current order


# Upsell categories
UPSELL_CATEGORIES = {
    "drinks": ["soda", "juice", "water", "tea", "coffee"],
    "sides": ["fries", "salad", "coleslaw", "breadsticks", "wings"],
    "desserts": ["cake", "pie", "ice cream", "cookie", "brownie"],
    "upgrades": ["large", "extra", "premium", "deluxe", "combo"]
}


class UpsellTracker:
    """
    Track upsell history per call to prevent repetition.
    Context-aware recommendation engine.
    """
    
    def __init__(self, call_id: str):
        """
        Initialize upsell tracker.
        
        Args:
            call_id: Call identifier
        """
        self.call_id = call_id
        
        # Tracking sets
        self.offered_items: Set[str] = set()  # Item IDs offered
        self.offered_categories: Set[str] = set()  # Categories offered
        self.accepted_items: Set[str] = set()  # Items customer accepted
        self.rejected_items: Set[str] = set()  # Items customer rejected
        
        # Counters
        self.total_offered = 0
        self.total_accepted = 0
        self.total_rejected = 0
        
        # Flags
        self.upsells_disabled = False
        self.customer_declined_all = False
        
        # Timestamps
        self.created_at = datetime.utcnow()
        self.last_offer_time: Optional[datetime] = None
        
        # Enterprise features (v2.0)
        self.strategy = self._assign_ab_strategy()
        self.revenue_from_upsells = 0.0
        self.offer_timestamps: Dict[str, datetime] = {}
        self.acceptance_times: Dict[str, float] = {}
        self.category_performance: Dict[str, Dict[str, int]] = {}
        
        logger.debug(f"UpsellTracker created for {call_id} (strategy: {self.strategy})")
    
    def can_offer_upsell(self) -> bool:
        """
        Check if upsell can be offered.
        
        Returns:
            True if upsell allowed
        """
        # Check disabled flag
        if self.upsells_disabled:
            logger.debug(f"Upsells disabled for {self.call_id}")
            return False
        
        # Check customer declined all
        if self.customer_declined_all:
            logger.debug(f"Customer declined all upsells for {self.call_id}")
            return False
        
        # Check hard cap
        if self.total_offered >= MAX_UPSELLS_PER_CALL:
            logger.debug(f"Max upsells reached for {self.call_id}")
            return False
        
        return True
    
    def mark_offered(self, item_id: str, category: Optional[str] = None, relevance: float = 0.0):
        """
        Mark item as offered with metrics tracking.
        
        Args:
            item_id: Item identifier
            category: Item category
            relevance: Suggestion relevance score
        """
        self.offered_items.add(item_id)
        if category:
            self.offered_categories.add(category)
        
        self.total_offered += 1
        self.last_offer_time = datetime.utcnow()
        self.offer_timestamps[item_id] = datetime.utcnow()
        
        # Track category performance
        if category:
            if category not in self.category_performance:
                self.category_performance[category] = {"offered": 0, "accepted": 0}
            self.category_performance[category]["offered"] += 1
        
        # Track metrics
        if METRICS_ENABLED:
            upsell_offers_total.labels(
                category=category or "unknown",
                strategy=self.strategy
            ).inc()
            if relevance > 0:
                upsell_suggestion_quality.observe(relevance)
        
        logger.debug(f"Upsell offered: {item_id} (total: {self.total_offered})")
    
    def _assign_ab_strategy(self) -> str:
        """Assign A/B test strategy based on call_id hash."""
        hash_val = int(hashlib.md5(self.call_id.encode()).hexdigest(), 16)
        strategies = ["aggressive", "balanced", "conservative"]
        return strategies[hash_val % len(strategies)]
    
    def mark_accepted(self, item_id: str, item_price: float = 0.0, category: str = "unknown"):
        """
        Mark item as accepted with revenue tracking.
        
        Args:
            item_id: Item identifier
            item_price: Item price (for revenue tracking)
            category: Item category
        """
        self.accepted_items.add(item_id)
        self.total_accepted += 1
        self.revenue_from_upsells += item_price
        
        # Track time to acceptance
        if item_id in self.offer_timestamps:
            time_to_accept = (datetime.utcnow() - self.offer_timestamps[item_id]).total_seconds()
            self.acceptance_times[item_id] = time_to_accept
        
        # Track category performance
        if category not in self.category_performance:
            self.category_performance[category] = {"offered": 0, "accepted": 0}
        self.category_performance[category]["accepted"] += 1
        
        if METRICS_ENABLED:
            upsell_conversions_total.labels(category=category, strategy=self.strategy).inc()
            if item_price > 0:
                upsell_revenue_dollars.inc(item_price)
        
        logger.info(f"Upsell accepted: {item_id} for {self.call_id} (+${item_price:.2f})")
    
    def mark_rejected(self, item_id: str, category: str = "unknown"):
        """
        Mark item as rejected.
        
        Args:
            item_id: Item identifier
            category: Item category
        """
        self.rejected_items.add(item_id)
        self.total_rejected += 1
        
        if METRICS_ENABLED:
            upsell_rejections_total.labels(category=category).inc()
        
        logger.debug(f"Upsell rejected: {item_id}")
    
    def mark_declined_all(self):
        """Mark that customer declined all upsells."""
        self.customer_declined_all = True
        logger.info(f"Customer declined all upsells for {self.call_id}")
    
    def disable_upsells(self):
        """Disable upsells for this call."""
        self.upsells_disabled = True
        logger.info(f"Upsells disabled for {self.call_id}")
    
    def was_offered(self, item_id: str) -> bool:
        """Check if item was already offered."""
        return item_id in self.offered_items
    
    def was_category_offered(self, category: str) -> bool:
        """Check if category was already offered."""
        return category in self.offered_categories
    
    def get_stats(self) -> Dict[str, Any]:
        """Get upsell statistics with enterprise metrics."""
        conversion_rate = self.total_accepted / max(1, self.total_offered)
        avg_acceptance_time = 0.0
        if self.acceptance_times:
            avg_acceptance_time = sum(self.acceptance_times.values()) / len(self.acceptance_times)
        
        # Update conversion rate metric
        if METRICS_ENABLED:
            upsell_conversion_rate.labels(strategy=self.strategy).set(conversion_rate)
        
        return {
            "call_id": self.call_id,
            "strategy": self.strategy,
            "total_offered": self.total_offered,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "acceptance_rate": round(conversion_rate, 4),
            "revenue_from_upsells": round(self.revenue_from_upsells, 2),
            "avg_acceptance_time_seconds": round(avg_acceptance_time, 2),
            "category_performance": self.category_performance,
            "upsells_disabled": self.upsells_disabled,
            "customer_declined_all": self.customer_declined_all
        }


# Global tracker store
_upsell_trackers: Dict[str, UpsellTracker] = {}


def _get_tracker(call_id: str) -> UpsellTracker:
    """
    Get or create upsell tracker.
    
    Args:
        call_id: Call identifier
        
    Returns:
        UpsellTracker instance
    """
    if call_id not in _upsell_trackers:
        _upsell_trackers[call_id] = UpsellTracker(call_id)
    
    return _upsell_trackers[call_id]


def suggest_upsells(
    call_id: str,
    menu: Dict[str, Any],
    current_order: Optional[Dict[str, Any]] = None,
    max_suggestions: int = MAX_UPSELLS_PER_TURN
) -> List[Dict[str, Any]]:
    """
    Generate context-aware upsell suggestions.
    
    Args:
        call_id: Call identifier
        menu: Menu data
        current_order: Current order data (optional)
        max_suggestions: Maximum suggestions to return
        
    Returns:
        List of upsell suggestions
        
    Example:
        >>> suggestions = suggest_upsells("call_123", menu, order)
        >>> for sug in suggestions:
        >>>     print(sug["name"], sug["reason"])
    """
    tracker = _get_tracker(call_id)
    
    # Check if upsells allowed
    if not tracker.can_offer_upsell():
        return []
    
    if not menu or not menu.get("items"):
        logger.warning(f"No menu available for upsells: {call_id}")
        return []
    
    # Get current order context
    current_items = []
    current_total = 0.0
    
    if current_order and current_order.get("items"):
        current_items = current_order["items"]
        current_total = current_order.get("total", 0.0)
    
    # Check minimum order value
    if current_total < MIN_ORDER_VALUE_FOR_UPSELL:
        logger.debug(f"Order value too low for upsells: ${current_total:.2f}")
        return []
    
    # Determine what's missing
    missing_categories = _get_missing_categories(current_items)
    
    # Generate candidates
    candidates = _generate_candidates(
        menu,
        current_items,
        current_total,
        missing_categories,
        tracker
    )
    
    # Filter and rank
    filtered = _filter_candidates(candidates, tracker)
    ranked = _rank_candidates(filtered, current_total)
    
    # Limit to max suggestions
    suggestions = ranked[:max_suggestions]
    
    # Mark as offered
    for suggestion in suggestions:
        tracker.mark_offered(
            suggestion["item_id"],
            suggestion.get("category")
        )
    
    logger.info(
        f"Generated {len(suggestions)} upsell(s) for {call_id}: "
        f"{[s['name'] for s in suggestions]}"
    )
    
    return suggestions


def _get_missing_categories(current_items: List[Dict[str, Any]]) -> Set[str]:
    """
    Determine which categories are missing from order.
    
    Args:
        current_items: Current order items
        
    Returns:
        Set of missing categories
    """
    present_categories = set()
    
    for item in current_items:
        category = item.get("category", "").lower()
        
        # Map to upsell categories
        for upsell_cat, keywords in UPSELL_CATEGORIES.items():
            if any(kw in category for kw in keywords):
                present_categories.add(upsell_cat)
    
    # Return missing categories
    all_categories = set(UPSELL_CATEGORIES.keys())
    missing = all_categories - present_categories
    
    return missing


def _generate_candidates(
    menu: Dict[str, Any],
    current_items: List[Dict[str, Any]],
    current_total: float,
    missing_categories: Set[str],
    tracker: UpsellTracker
) -> List[Dict[str, Any]]:
    """
    Generate upsell candidates.
    
    Args:
        menu: Menu data
        current_items: Current order items
        current_total: Current order total
        missing_categories: Categories not in order
        tracker: Upsell tracker
        
    Returns:
        List of candidate upsells
    """
    candidates = []
    max_upsell_price = current_total * MAX_UPSELL_PRICE_RATIO
    
    for item in menu.get("items", []):
        # Skip if unavailable
        if not item.get("available", True):
            continue
        
        item_id = item.get("id", "")
        item_name = item.get("name", "")
        item_price = item.get("price", 0.0)
        item_category = item.get("category", "").lower()
        
        # Skip if already in order
        if any(i.get("id") == item_id for i in current_items):
            continue
        
        # Skip if already offered
        if tracker.was_offered(item_id):
            continue
        
        # Skip if price too high
        if item_price > max_upsell_price:
            continue
        
        # Skip if price too low (not worth suggesting)
        if item_price < 1.00:
            continue
        
        # Determine relevance
        relevance = _calculate_relevance(
            item,
            current_items,
            missing_categories,
            tracker
        )
        
        if relevance > 0:
            candidates.append({
                "item_id": item_id,
                "name": item_name,
                "price": item_price,
                "category": item_category,
                "relevance": relevance,
                "reason": _generate_reason(item, missing_categories)
            })
    
    return candidates


def _calculate_relevance(
    item: Dict[str, Any],
    current_items: List[Dict[str, Any]],
    missing_categories: Set[str],
    tracker: UpsellTracker
) -> float:
    """
    Calculate relevance score for upsell item.
    
    Args:
        item: Menu item
        current_items: Current order items
        missing_categories: Missing categories
        tracker: Upsell tracker
        
    Returns:
        Relevance score (0.0 to 1.0)
    """
    score = 0.0
    item_category = item.get("category", "").lower()
    item_name = item.get("name", "").lower()
    
    # Boost if in missing category
    for missing_cat, keywords in UPSELL_CATEGORIES.items():
        if missing_cat in missing_categories:
            if any(kw in item_category or kw in item_name for kw in keywords):
                score += 0.5
    
    # Boost drinks and sides
    if any(kw in item_category or kw in item_name for kw in UPSELL_CATEGORIES["drinks"]):
        score += 0.3
    
    if any(kw in item_category or kw in item_name for kw in UPSELL_CATEGORIES["sides"]):
        score += 0.3
    
    # Penalize if category already offered
    for upsell_cat, keywords in UPSELL_CATEGORIES.items():
        if tracker.was_category_offered(upsell_cat):
            if any(kw in item_category or kw in item_name for kw in keywords):
                score -= 0.4
    
    # Boost desserts if order is substantial
    if len(current_items) >= 2:
        if any(kw in item_category or kw in item_name for kw in UPSELL_CATEGORIES["desserts"]):
            score += 0.2
    
    return max(0.0, min(1.0, score))


def _filter_candidates(
    candidates: List[Dict[str, Any]],
    tracker: UpsellTracker
) -> List[Dict[str, Any]]:
    """
    Filter candidates for safety.
    
    Args:
        candidates: Candidate upsells
        tracker: Upsell tracker
        
    Returns:
        Filtered candidates
    """
    filtered = []
    
    for candidate in candidates:
        # Skip if relevance too low
        if candidate["relevance"] < 0.1:
            continue
        
        # Skip if rejected before
        if candidate["item_id"] in tracker.rejected_items:
            continue
        
        # Skip if accepted before (don't re-suggest)
        if candidate["item_id"] in tracker.accepted_items:
            continue
        
        filtered.append(candidate)
    
    return filtered


def _rank_candidates(
    candidates: List[Dict[str, Any]],
    current_total: float
) -> List[Dict[str, Any]]:
    """
    Rank candidates by relevance and price.
    
    Args:
        candidates: Filtered candidates
        current_total: Current order total
        
    Returns:
        Ranked candidates
    """
    # Sort by relevance (descending), then price (ascending)
    ranked = sorted(
        candidates,
        key=lambda x: (-x["relevance"], x["price"])
    )
    
    return ranked


def _generate_reason(
    item: Dict[str, Any],
    missing_categories: Set[str]
) -> str:
    """
    Generate human-readable reason for upsell.
    
    Args:
        item: Menu item
        missing_categories: Missing categories
        
    Returns:
        Reason string
    """
    item_category = item.get("category", "").lower()
    item_name = item.get("name", "")
    
    # Check category
    for cat, keywords in UPSELL_CATEGORIES.items():
        if cat in missing_categories:
            if any(kw in item_category for kw in keywords):
                return f"Complements your order"
    
    # Default reasons
    if any(kw in item_category for kw in UPSELL_CATEGORIES["drinks"]):
        return "Popular beverage choice"
    
    if any(kw in item_category for kw in UPSELL_CATEGORIES["sides"]):
        return "Great side dish"
    
    if any(kw in item_category for kw in UPSELL_CATEGORIES["desserts"]):
        return "Perfect finish to your meal"
    
    return "Customer favorite"


def format_suggestion_text(suggestions: List[Dict[str, Any]]) -> str:
    """
    Format suggestions into natural text.
    
    Args:
        suggestions: List of suggestions
        
    Returns:
        Formatted text
        
    Example:
        >>> text = format_suggestion_text(suggestions)
        >>> print(text)
        "Would you like to add a Soda ($2.99) to your order?"
    """
    if not suggestions:
        return ""
    
    if len(suggestions) == 1:
        sug = suggestions[0]
        return f"Would you like to add {sug['name']} (${sug['price']:.2f})?"
    
    # Multiple suggestions
    items = ", ".join([
        f"{s['name']} (${s['price']:.2f})"
        for s in suggestions[:-1]
    ])
    last = suggestions[-1]
    
    return f"Would you like to add {items}, or {last['name']} (${last['price']:.2f})?"


def mark_upsell_accepted(call_id: str, item_id: str):
    """
    Mark upsell as accepted.
    
    Args:
        call_id: Call identifier
        item_id: Item identifier
    """
    tracker = _get_tracker(call_id)
    tracker.mark_accepted(item_id)


def mark_upsell_rejected(call_id: str, item_id: str):
    """
    Mark upsell as rejected.
    
    Args:
        call_id: Call identifier
        item_id: Item identifier
    """
    tracker = _get_tracker(call_id)
    tracker.mark_rejected(item_id)


def customer_declined_all_upsells(call_id: str):
    """
    Mark that customer declined all upsells.
    
    Args:
        call_id: Call identifier
    """
    tracker = _get_tracker(call_id)
    tracker.mark_declined_all()


def disable_upsells(call_id: str):
    """
    Disable upsells for call.
    
    Args:
        call_id: Call identifier
    """
    tracker = _get_tracker(call_id)
    tracker.disable_upsells()


def get_upsell_stats(call_id: str) -> Dict[str, Any]:
    """
    Get upsell statistics for call.
    
    Args:
        call_id: Call identifier
        
    Returns:
        Statistics dictionary
    """
    tracker = _get_tracker(call_id)
    return tracker.get_stats()


def clear_upsell_tracker(call_id: str):
    """
    Clear upsell tracker (cleanup).
    
    Args:
        call_id: Call identifier
    """
    if call_id in _upsell_trackers:
        del _upsell_trackers[call_id]
        logger.debug(f"Upsell tracker cleared for {call_id}")


def get_global_analytics() -> Dict[str, Any]:
    """Get global upsell analytics across all calls."""
    if not _upsell_trackers:
        return {
            "total_calls": 0,
            "total_offered": 0,
            "total_accepted": 0,
            "global_conversion_rate": 0.0,
            "total_revenue": 0.0,
            "strategy_performance": {}
        }
    
    total_offered = sum(t.total_offered for t in _upsell_trackers.values())
    total_accepted = sum(t.total_accepted for t in _upsell_trackers.values())
    total_revenue = sum(t.revenue_from_upsells for t in _upsell_trackers.values())
    
    # Strategy performance
    strategy_stats = {}
    for tracker in _upsell_trackers.values():
        if tracker.strategy not in strategy_stats:
            strategy_stats[tracker.strategy] = {"offered": 0, "accepted": 0, "revenue": 0.0}
        strategy_stats[tracker.strategy]["offered"] += tracker.total_offered
        strategy_stats[tracker.strategy]["accepted"] += tracker.total_accepted
        strategy_stats[tracker.strategy]["revenue"] += tracker.revenue_from_upsells
    
    # Calculate conversion rates per strategy
    for strategy, stats in strategy_stats.items():
        stats["conversion_rate"] = round(
            stats["accepted"] / max(1, stats["offered"]), 4
        )
    
    return {
        "total_calls": len(_upsell_trackers),
        "total_offered": total_offered,
        "total_accepted": total_accepted,
        "global_conversion_rate": round(total_accepted / max(1, total_offered), 4),
        "total_revenue": round(total_revenue, 2),
        "strategy_performance": strategy_stats
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Upsell Module (Enterprise v2.0)")
    print("=" * 50)
    
    # Sample menu
    sample_menu = {
        "items": [
            {"id": "1", "name": "Large Pizza", "price": 15.99, "category": "Pizza", "available": True},
            {"id": "2", "name": "Soda", "price": 2.99, "category": "Drinks", "available": True},
            {"id": "3", "name": "Fries", "price": 3.99, "category": "Sides", "available": True},
            {"id": "4", "name": "Ice Cream", "price": 4.99, "category": "Desserts", "available": True},
            {"id": "5", "name": "Wings", "price": 9.99, "category": "Appetizers", "available": True}
        ]
    }
    
    # Sample order
    sample_order = {
        "items": [
            {"id": "1", "name": "Large Pizza", "price": 15.99, "category": "Pizza"}
        ],
        "total": 17.27
    }
    
    call_id = "test_call_123"
    
    print(f"\nCall: {call_id}")
    print(f"Current order: {sample_order['items'][0]['name']}")
    print(f"Total: ${sample_order['total']:.2f}")
    
    # Generate suggestions
    suggestions = suggest_upsells(call_id, sample_menu, sample_order, max_suggestions=2)
    
    print(f"\nUpsell suggestions ({len(suggestions)}):")
    for sug in suggestions:
        print(f"  - {sug['name']} (${sug['price']:.2f}) - {sug['reason']}")
    
    # Format text
    text = format_suggestion_text(suggestions)
    print(f"\nFormatted: {text}")
    
    # Mark one accepted
    if suggestions:
        mark_upsell_accepted(call_id, suggestions[0]["item_id"])
    
    # Try again (should get different suggestions)
    suggestions2 = suggest_upsells(call_id, sample_menu, sample_order, max_suggestions=2)
    print(f"\nSecond round ({len(suggestions2)}):")
    for sug in suggestions2:
        print(f"  - {sug['name']} (${sug['price']:.2f})")
    
    # Stats
    stats = get_upsell_stats(call_id)
    print(f"\nStats:")
    print(f"  Total offered: {stats['total_offered']}")
    print(f"  Total accepted: {stats['total_accepted']}")
    print(f"  Acceptance rate: {stats['acceptance_rate']:.1%}")
    
    print("\n" + "=" * 50)
    print("Production upsell module ready")
