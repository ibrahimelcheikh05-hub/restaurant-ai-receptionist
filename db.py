"""
Database Module (Production)
=============================
Hardened async database layer with Supabase.
Retry queues, fire-and-forget writes, circuit breakers, failure tolerance.
Never blocks live calls.
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import json

from supabase import create_client, Client
from postgrest.exceptions import APIError


logger = logging.getLogger(__name__)


# Configuration
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_TIMEOUT = 30  # seconds
MAX_WRITE_QUEUE_SIZE = 1000
BATCH_WRITE_SIZE = 10
BATCH_WRITE_INTERVAL = 2.0  # seconds


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"    # Normal operation
    OPEN = "open"        # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker for database operations."""
    
    def __init__(
        self,
        threshold: int = CIRCUIT_BREAKER_THRESHOLD,
        timeout: int = CIRCUIT_BREAKER_TIMEOUT
    ):
        self.threshold = threshold
        self.timeout = timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.success_count = 0
    
    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 2:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                logger.info("Circuit breaker closed (recovered)")
    
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.threshold:
            self.state = CircuitState.OPEN
            logger.error(
                f"Circuit breaker opened "
                f"(failures: {self.failure_count})"
            )
    
    def can_execute(self) -> bool:
        """Check if operation can execute."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if timeout expired
            if self.last_failure_time:
                elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
                if elapsed >= self.timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info("Circuit breaker half-open (testing)")
                    return True
            return False
        
        # HALF_OPEN - allow one test request
        return True
    
    def get_state(self) -> str:
        """Get current state."""
        return self.state.value


class WriteQueue:
    """Fire-and-forget write queue with batching."""
    
    def __init__(self, max_size: int = MAX_WRITE_QUEUE_SIZE):
        self.queue: deque = deque(maxlen=max_size)
        self.max_size = max_size
        self.dropped_count = 0
    
    def enqueue(self, operation: Dict[str, Any]) -> bool:
        """
        Enqueue write operation.
        
        Args:
            operation: Write operation data
            
        Returns:
            True if enqueued, False if queue full
        """
        if len(self.queue) >= self.max_size:
            self.dropped_count += 1
            logger.warning(
                f"Write queue full, dropping write "
                f"(dropped: {self.dropped_count})"
            )
            return False
        
        self.queue.append(operation)
        return True
    
    def dequeue_batch(self, size: int) -> List[Dict[str, Any]]:
        """Dequeue batch of operations."""
        batch = []
        for _ in range(min(size, len(self.queue))):
            if self.queue:
                batch.append(self.queue.popleft())
        return batch
    
    def size(self) -> int:
        """Get queue size."""
        return len(self.queue)
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self.queue) == 0


class DatabaseClient:
    """
    Production database client with resilience features.
    Async-only, fire-and-forget writes, circuit breakers.
    """
    
    def __init__(self):
        self.client: Optional[Client] = None
        self.write_queue = WriteQueue()
        self.circuit_breaker = CircuitBreaker()
        
        # Background tasks
        self.write_processor_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Stats
        self.read_count = 0
        self.write_count = 0
        self.error_count = 0
        self.retry_count = 0
        
        # Initialize client
        self._initialize_client()
        
        logger.info("DatabaseClient initialized")
    
    def _initialize_client(self):
        """Initialize Supabase client."""
        try:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")
            
            if not url or not key:
                logger.error("SUPABASE_URL and SUPABASE_KEY required")
                return
            
            self.client = create_client(url, key)
            logger.info("Supabase client initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
    
    async def start(self):
        """Start background write processor."""
        if self.is_running:
            return
        
        self.is_running = True
        self.write_processor_task = asyncio.create_task(
            self._write_processor_loop()
        )
        logger.info("Database write processor started")
    
    async def stop(self):
        """Stop background write processor."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.write_processor_task and not self.write_processor_task.done():
            self.write_processor_task.cancel()
            try:
                await self.write_processor_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining writes
        await self._flush_writes()
        
        logger.info("Database write processor stopped")
    
    async def _write_processor_loop(self):
        """Background loop to process write queue."""
        try:
            while self.is_running:
                await asyncio.sleep(BATCH_WRITE_INTERVAL)
                
                if self.write_queue.is_empty():
                    continue
                
                await self._process_write_batch()
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Write processor error: {str(e)}")
    
    async def _process_write_batch(self):
        """Process batch of writes."""
        batch = self.write_queue.dequeue_batch(BATCH_WRITE_SIZE)
        
        if not batch:
            return
        
        logger.debug(f"Processing write batch: {len(batch)} operations")
        
        for operation in batch:
            try:
                await self._execute_write(operation)
            except Exception as e:
                logger.error(f"Batch write error: {str(e)}")
                # Continue with next operation (fire-and-forget)
    
    async def _execute_write(self, operation: Dict[str, Any]):
        """Execute single write operation with retry."""
        table = operation.get("table")
        data = operation.get("data")
        op_type = operation.get("type", "insert")
        
        if not self.client or not table or not data:
            return
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                loop = asyncio.get_event_loop()
                
                if op_type == "insert":
                    await loop.run_in_executor(
                        None,
                        lambda: self.client.table(table).insert(data).execute()
                    )
                elif op_type == "upsert":
                    await loop.run_in_executor(
                        None,
                        lambda: self.client.table(table).upsert(data).execute()
                    )
                
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
        """Flush all pending writes."""
        logger.info(f"Flushing {self.write_queue.size()} pending writes")
        
        while not self.write_queue.is_empty():
            await self._process_write_batch()
    
    # ========================================================================
    # READ OPERATIONS (Async with timeout and circuit breaker)
    # ========================================================================
    
    async def fetch_menu(
        self,
        restaurant_id: str,
        timeout: float = 5.0
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch menu (async, with timeout).
        
        Args:
            restaurant_id: Restaurant identifier
            timeout: Operation timeout
            
        Returns:
            Menu data or None on error
        """
        if not self.client:
            logger.error("Database client not initialized")
            return None
        
        if not self.circuit_breaker.can_execute():
            logger.warning("Circuit breaker open, skipping read")
            return None
        
        try:
            loop = asyncio.get_event_loop()
            
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.client
                        .table("menus")
                        .select("*")
                        .eq("restaurant_id", restaurant_id)
                        .execute()
                ),
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
    
    # ========================================================================
    # WRITE OPERATIONS (Fire-and-forget)
    # ========================================================================
    
    def store_call_log(self, call_data: Dict[str, Any]) -> bool:
        """
        Store call log (fire-and-forget).
        
        Args:
            call_data: Call log data
            
        Returns:
            True if enqueued
        """
        operation = {
            "type": "insert",
            "table": "call_logs",
            "data": {
                **call_data,
                "created_at": datetime.utcnow().isoformat()
            }
        }
        
        return self.write_queue.enqueue(operation)
    
    def store_order(self, order_data: Dict[str, Any]) -> bool:
        """
        Store order (fire-and-forget).
        
        Args:
            order_data: Order data
            
        Returns:
            True if enqueued
        """
        operation = {
            "type": "insert",
            "table": "orders",
            "data": {
                **order_data,
                "created_at": datetime.utcnow().isoformat()
            }
        }
        
        return self.write_queue.enqueue(operation)
    
    def update_order_status(
        self,
        order_id: str,
        status: str
    ) -> bool:
        """
        Update order status (fire-and-forget).
        
        Args:
            order_id: Order identifier
            status: New status
            
        Returns:
            True if enqueued
        """
        operation = {
            "type": "upsert",
            "table": "orders",
            "data": {
                "order_id": order_id,
                "status": status,
                "updated_at": datetime.utcnow().isoformat()
            }
        }
        
        return self.write_queue.enqueue(operation)
    
    def store_transcript(
        self,
        call_id: str,
        transcript_data: Dict[str, Any]
    ) -> bool:
        """
        Store conversation transcript (fire-and-forget).
        
        Args:
            call_id: Call identifier
            transcript_data: Transcript data
            
        Returns:
            True if enqueued
        """
        operation = {
            "type": "insert",
            "table": "transcripts",
            "data": {
                "call_id": call_id,
                **transcript_data,
                "created_at": datetime.utcnow().isoformat()
            }
        }
        
        return self.write_queue.enqueue(operation)
    
    # ========================================================================
    # STATS & MONITORING
    # ========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            "reads": self.read_count,
            "writes": self.write_count,
            "errors": self.error_count,
            "retries": self.retry_count,
            "queue_size": self.write_queue.size(),
            "queue_dropped": self.write_queue.dropped_count,
            "circuit_breaker": self.circuit_breaker.get_state(),
            "circuit_failures": self.circuit_breaker.failure_count
        }
    
    def get_queue_size(self) -> int:
        """Get write queue size."""
        return self.write_queue.size()
    
    def is_healthy(self) -> bool:
        """Check if database is healthy."""
        return (
            self.client is not None and
            self.circuit_breaker.state != CircuitState.OPEN
        )


# ============================================================================
# GLOBAL DATABASE INSTANCE
# ============================================================================

db = DatabaseClient()


# ============================================================================
# AUTO-START WRITE PROCESSOR
# ============================================================================

async def _auto_start():
    """Auto-start write processor."""
    await db.start()


# Initialize on import (non-blocking)
try:
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.create_task(_auto_start())
    else:
        loop.run_until_complete(_auto_start())
except RuntimeError:
    # No event loop, will start on first use
    pass


# ============================================================================
# GRACEFUL SHUTDOWN
# ============================================================================

async def shutdown_db():
    """Graceful database shutdown."""
    await db.stop()
    logger.info("Database shutdown complete")


# ============================================================================
# HEALTH CHECK
# ============================================================================

def is_db_healthy() -> bool:
    """Check database health."""
    return db.is_healthy()


def get_db_stats() -> Dict[str, Any]:
    """Get database statistics."""
    return db.get_stats()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def example():
        print("Database Module (Production)")
        print("=" * 50)
        
        # Start processor
        await db.start()
        print("\nWrite processor started")
        
        # Fire-and-forget writes
        print("\nEnqueuing writes...")
        db.store_call_log({
            "restaurant_id": "rest_001",
            "call_sid": "call_123",
            "status": "completed"
        })
        
        db.store_order({
            "order_id": "ord_123",
            "restaurant_id": "rest_001",
            "total": 25.99
        })
        
        print(f"Queue size: {db.get_queue_size()}")
        
        # Stats
        print("\nDatabase stats:")
        stats = get_db_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Health check
        print(f"\nHealthy: {is_db_healthy()}")
        
        # Wait for writes to process
        await asyncio.sleep(3)
        
        # Shutdown
        await shutdown_db()
        print("\nDatabase shutdown complete")
        
        print("\n" + "=" * 50)
        print("Production database module ready")
    
    asyncio.run(example())
