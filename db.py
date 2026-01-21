"""
Database Module (Enterprise Production)
========================================
Enterprise-grade async database layer with Supabase.

NEW FEATURES (Enterprise v2.0):
✅ Health check monitoring (every 10s)
✅ Prometheus metrics integration
✅ Connection validation
✅ Detailed health status API
✅ Operation tracking decorator
✅ Enhanced error logging
✅ Configuration via environment variables
✅ URL validation (HTTPS enforcement)

Version: 2.0.0 (Enterprise)
Last Updated: 2026-01-21
Security: Reviewed - No SQL injection risks (uses Supabase ORM)
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import deque
from enum import Enum
import time
from functools import wraps

try:
    from supabase import create_client, Client
    from postgrest.exceptions import APIError
except ImportError:
    Client = None
    APIError = Exception

try:
    from prometheus_client import Counter, Histogram, Gauge
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False


logger = logging.getLogger(__name__)


# Configuration from environment
MAX_RETRIES = int(os.getenv("MAX_RETRIES_DB", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY_SECONDS", "1.0"))
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))
CIRCUIT_BREAKER_TIMEOUT = int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "30"))
MAX_WRITE_QUEUE_SIZE = int(os.getenv("MAX_WRITE_QUEUE_SIZE", "1000"))
BATCH_WRITE_SIZE = 10
BATCH_WRITE_INTERVAL = 2.0
DB_HEALTH_CHECK_INTERVAL = 10


# Prometheus metrics
if METRICS_ENABLED:
    db_operations_total = Counter(
        'db_operations_total',
        'Total database operations',
        ['operation', 'status']
    )
    db_operation_duration = Histogram(
        'db_operation_duration_seconds',
        'Database operation duration',
        ['operation']
    )
    db_queue_size_metric = Gauge(
        'db_write_queue_size',
        'Current write queue size'
    )
    db_circuit_state_metric = Gauge(
        'db_circuit_breaker_state',
        'Circuit breaker state'
    )


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker with metrics."""
    
    def __init__(self, threshold: int = CIRCUIT_BREAKER_THRESHOLD, timeout: int = CIRCUIT_BREAKER_TIMEOUT):
        self.threshold = threshold
        self.timeout = timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.success_count = 0
        self._update_metrics()
    
    def _update_metrics(self):
        if METRICS_ENABLED:
            state_map = {CircuitState.CLOSED: 0, CircuitState.OPEN: 1, CircuitState.HALF_OPEN: 2}
            db_circuit_state_metric.set(state_map[self.state])
    
    def record_success(self):
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 2:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                logger.info("Circuit breaker closed (recovered)")
                self._update_metrics()
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        if self.failure_count >= self.threshold:
            old_state = self.state
            self.state = CircuitState.OPEN
            if old_state != CircuitState.OPEN:
                logger.error(f"Circuit breaker opened (failures: {self.failure_count})")
                self._update_metrics()
    
    def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if self.last_failure_time:
                elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
                if elapsed >= self.timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info("Circuit breaker half-open (testing)")
                    self._update_metrics()
                    return True
            return False
        return True
    
    def get_state(self) -> str:
        return self.state.value
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class WriteQueue:
    """Write queue with metrics."""
    
    def __init__(self, max_size: int = MAX_WRITE_QUEUE_SIZE):
        self.queue: deque = deque(maxlen=max_size)
        self.max_size = max_size
        self.dropped_count = 0
        self._update_metrics()
    
    def _update_metrics(self):
        if METRICS_ENABLED:
            db_queue_size_metric.set(len(self.queue))
    
    def enqueue(self, operation: Dict[str, Any]) -> bool:
        if len(self.queue) >= self.max_size:
            self.dropped_count += 1
            logger.warning(f"Write queue full, dropping write (dropped: {self.dropped_count})")
            if METRICS_ENABLED:
                db_operations_total.labels(operation='write', status='dropped').inc()
            return False
        self.queue.append(operation)
        self._update_metrics()
        return True
    
    def dequeue_batch(self, size: int) -> List[Dict[str, Any]]:
        batch = []
        for _ in range(min(size, len(self.queue))):
            if self.queue:
                batch.append(self.queue.popleft())
        self._update_metrics()
        return batch
    
    def size(self) -> int:
        return len(self.queue)
    
    def is_empty(self) -> bool:
        return len(self.queue) == 0


def track_operation(operation_name: str):
    """Decorator to track operations."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                if METRICS_ENABLED:
                    db_operations_total.labels(operation=operation_name, status='success').inc()
                    db_operation_duration.labels(operation=operation_name).observe(duration)
                logger.debug(f"{operation_name} completed in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                if METRICS_ENABLED:
                    db_operations_total.labels(operation=operation_name, status='error').inc()
                logger.error(f"{operation_name} failed after {duration:.3f}s: {str(e)}")
                raise
        return wrapper
    return decorator


class DatabaseClient:
    """Enterprise database client with full observability."""
    
    def __init__(self):
        self.client: Optional[Client] = None
        self.write_queue = WriteQueue()
        self.circuit_breaker = CircuitBreaker()
        self.write_processor_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        self.is_running = False
        self.last_health_check: Optional[datetime] = None
        self.health_status = "unknown"
        self.consecutive_health_failures = 0
        self.read_count = 0
        self.write_count = 0
        self.error_count = 0
        self.retry_count = 0
        self.uptime_start = datetime.utcnow()
        self._initialize_client()
        logger.info("DatabaseClient initialized (Enterprise v2.0)")
    
    def _initialize_client(self):
        """Initialize Supabase client with validation."""
        try:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")
            
            if not url or not key:
                logger.error("SUPABASE_URL and SUPABASE_KEY required")
                self.health_status = "unhealthy"
                return
            
            if not url.startswith("https://"):
                logger.error(f"Invalid Supabase URL (must use HTTPS): {url}")
                self.health_status = "unhealthy"
                return
            
            self.client = create_client(url, key)
            self.health_status = "healthy"
            logger.info(f"Supabase client initialized: {url[:30]}...")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
            self.health_status = "unhealthy"
    
    async def start(self):
        """Start background processors."""
        if self.is_running:
            return
        self.is_running = True
        self.write_processor_task = asyncio.create_task(self._write_processor_loop())
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Database background tasks started")
    
    async def stop(self):
        """Stop background processors."""
        if not self.is_running:
            return
        self.is_running = False
        for task in [self.write_processor_task, self.health_check_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await self._flush_writes()
        logger.info("Database background tasks stopped")
    
    async def _write_processor_loop(self):
        """Background write processor."""
        try:
            while self.is_running:
                await asyncio.sleep(BATCH_WRITE_INTERVAL)
                if not self.write_queue.is_empty():
                    await self._process_write_batch()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Write processor error: {str(e)}")
    
    async def _health_check_loop(self):
        """Background health check."""
        try:
            while self.is_running:
                await asyncio.sleep(DB_HEALTH_CHECK_INTERVAL)
                await self._perform_health_check()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
    
    async def _perform_health_check(self):
        """Perform database health check."""
        try:
            if not self.client:
                self.health_status = "unhealthy"
                self.consecutive_health_failures += 1
                return
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self.client.table("call_logs").select("*").limit(1).execute()),
                timeout=5.0
            )
            self.health_status = "healthy"
            self.consecutive_health_failures = 0
            self.last_health_check = datetime.utcnow()
        except asyncio.TimeoutError:
            self.health_status = "degraded"
            self.consecutive_health_failures += 1
            logger.warning("Database health check timeout")
        except Exception as e:
            self.health_status = "unhealthy"
            self.consecutive_health_failures += 1
            logger.error(f"Database health check failed: {str(e)}")
        if self.consecutive_health_failures >= 3:
            logger.error(f"Database unhealthy for {self.consecutive_health_failures} consecutive checks")
    
    async def _process_write_batch(self):
        """Process write batch."""
        batch = self.write_queue.dequeue_batch(BATCH_WRITE_SIZE)
        if not batch:
            return
        logger.debug(f"Processing write batch: {len(batch)} operations")
        for operation in batch:
            try:
                await self._execute_write(operation)
            except Exception as e:
                logger.error(f"Batch write error: {str(e)}")
    
    @track_operation("write")
    async def _execute_write(self, operation: Dict[str, Any]):
        """Execute write with retry."""
        table = operation.get("table")
        data = operation.get("data")
        op_type = operation.get("type", "insert")
        if not self.client or not table or not data:
            return
        for attempt in range(MAX_RETRIES + 1):
            try:
                loop = asyncio.get_event_loop()
                if op_type == "insert":
                    await loop.run_in_executor(None, lambda: self.client.table(table).insert(data).execute())
                elif op_type == "upsert":
                    await loop.run_in_executor(None, lambda: self.client.table(table).upsert(data).execute())
                self.write_count += 1
                self.circuit_breaker.record_success()
                return
            except Exception as e:
                logger.error(f"Write error (attempt {attempt + 1}): {str(e)}")
                self.error_count += 1
                if attempt < MAX_RETRIES:
                    self.retry_count += 1
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    self.circuit_breaker.record_failure()
    
    async def _flush_writes(self):
        """Flush pending writes."""
        pending = self.write_queue.size()
        if pending > 0:
            logger.info(f"Flushing {pending} pending writes")
            while not self.write_queue.is_empty():
                await self._process_write_batch()
    
    @track_operation("fetch_menu")
    async def fetch_menu(self, restaurant_id: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Fetch menu with timeout."""
        if not self.client:
            logger.error("Database client not initialized")
            return None
        if not self.circuit_breaker.can_execute():
            logger.warning("Circuit breaker open, skipping read")
            return None
        try:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self.client.table("menus").select("*").eq("restaurant_id", restaurant_id).execute()),
                timeout=timeout
            )
            self.read_count += 1
            self.circuit_breaker.record_success()
            if result.data and len(result.data) > 0:
                return result.data[0]
            return None
        except asyncio.TimeoutError:
            logger.error(f"Menu fetch timeout for {restaurant_id}")
            self.error_count += 1
            self.circuit_breaker.record_failure()
            return None
        except Exception as e:
            logger.error(f"Menu fetch error: {str(e)}")
            self.error_count += 1
            self.circuit_breaker.record_failure()
            return None
    
    def store_call_log(self, call_data: Dict[str, Any]) -> bool:
        """Store call log (fire-and-forget)."""
        return self.write_queue.enqueue({
            "type": "insert",
            "table": "call_logs",
            "data": {**call_data, "created_at": datetime.utcnow().isoformat()}
        })
    
    def store_order(self, order_data: Dict[str, Any]) -> bool:
        """Store order (fire-and-forget)."""
        return self.write_queue.enqueue({
            "type": "insert",
            "table": "orders",
            "data": {**order_data, "created_at": datetime.utcnow().isoformat()}
        })
    
    def update_order_status(self, order_id: str, status: str) -> bool:
        """Update order status (fire-and-forget)."""
        return self.write_queue.enqueue({
            "type": "upsert",
            "table": "orders",
            "data": {"order_id": order_id, "status": status, "updated_at": datetime.utcnow().isoformat()}
        })
    
    def store_transcript(self, call_id: str, transcript_data: Dict[str, Any]) -> bool:
        """Store transcript (fire-and-forget)."""
        return self.write_queue.enqueue({
            "type": "insert",
            "table": "transcripts",
            "data": {"call_id": call_id, **transcript_data, "created_at": datetime.utcnow().isoformat()}
        })
    
    def is_healthy(self) -> bool:
        """Check health."""
        return (
            self.client is not None and
            self.circuit_breaker.state != CircuitState.OPEN and
            self.health_status in ["healthy", "degraded"]
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get detailed health status."""
        return {
            "status": self.health_status,
            "is_healthy": self.is_healthy(),
            "client_initialized": self.client is not None,
            "circuit_breaker": self.circuit_breaker.get_stats(),
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "consecutive_failures": self.consecutive_health_failures,
            "uptime_seconds": (datetime.utcnow() - self.uptime_start).total_seconds()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            "reads": self.read_count,
            "writes": self.write_count,
            "errors": self.error_count,
            "retries": self.retry_count,
            "queue_size": self.write_queue.size(),
            "queue_dropped": self.write_queue.dropped_count,
            "circuit_breaker": self.circuit_breaker.get_state(),
            "circuit_failures": self.circuit_breaker.failure_count,
            "health_status": self.health_status
        }
    
    def get_queue_size(self) -> int:
        """Get queue size."""
        return self.write_queue.size()


# Global instance
db = DatabaseClient()


# Auto-start
async def _auto_start():
    await db.start()

try:
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.create_task(_auto_start())
    else:
        loop.run_until_complete(_auto_start())
except RuntimeError:
    pass


# Public API
async def shutdown_db():
    """Graceful shutdown."""
    await db.stop()
    logger.info("Database shutdown complete")


def is_db_healthy() -> bool:
    """Check health."""
    return db.is_healthy()


def get_db_health() -> Dict[str, Any]:
    """Get health status."""
    return db.get_health_status()


def get_db_stats() -> Dict[str, Any]:
    """Get statistics."""
    return db.get_stats()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    async def example():
        print("Database Module (Enterprise v2.0)")
        print("="*50)
        await db.start()
        print("\nHealth:", get_db_health())
        print("\nStats:", get_db_stats())
        await asyncio.sleep(3)
        await shutdown_db()
    
    asyncio.run(example())
