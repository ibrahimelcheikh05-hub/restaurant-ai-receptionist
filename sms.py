"""
SMS Module (Production)
========================
Hardened SMS delivery with Twilio.
Background sending, retry logic, failure logging, idempotency protection.
Never blocks live calls.
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional, Set
from datetime import datetime
from collections import deque
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException


logger = logging.getLogger(__name__)


# Configuration
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds
MAX_SMS_QUEUE_SIZE = 500
SMS_PROCESSING_INTERVAL = 1.0  # seconds
MESSAGE_DEDUPLICATION_TTL = 3600  # 1 hour


class SMSMessage:
    """SMS message with metadata."""
    
    def __init__(
        self,
        to_number: str,
        message: str,
        message_id: Optional[str] = None
    ):
        self.to_number = to_number
        self.message = message
        self.message_id = message_id or self._generate_id()
        self.attempts = 0
        self.created_at = datetime.utcnow()
        self.last_attempt: Optional[datetime] = None
        self.error: Optional[str] = None
    
    def _generate_id(self) -> str:
        """Generate unique message ID for idempotency."""
        import hashlib
        content = f"{self.to_number}:{self.message}:{self.created_at.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "to_number": self.to_number,
            "message": self.message,
            "attempts": self.attempts,
            "created_at": self.created_at.isoformat(),
            "last_attempt": self.last_attempt.isoformat() if self.last_attempt else None,
            "error": self.error
        }


class SMSQueue:
    """Queue for background SMS sending."""
    
    def __init__(self, max_size: int = MAX_SMS_QUEUE_SIZE):
        self.queue: deque = deque(maxlen=max_size)
        self.max_size = max_size
        self.dropped_count = 0
        self.sent_ids: Set[str] = set()  # For deduplication
        self.sent_count = 0
        self.failed_count = 0
    
    def enqueue(self, message: SMSMessage) -> bool:
        """
        Enqueue SMS for sending.
        
        Args:
            message: SMS message
            
        Returns:
            True if enqueued, False if duplicate or queue full
        """
        # Check for duplicate
        if message.message_id in self.sent_ids:
            logger.debug(f"Duplicate SMS ignored: {message.message_id}")
            return False
        
        # Check queue capacity
        if len(self.queue) >= self.max_size:
            self.dropped_count += 1
            logger.warning(
                f"SMS queue full, dropping message "
                f"(dropped: {self.dropped_count})"
            )
            return False
        
        self.queue.append(message)
        logger.debug(f"SMS enqueued: {message.message_id} to {message.to_number}")
        return True
    
    def dequeue(self) -> Optional[SMSMessage]:
        """Dequeue SMS for sending."""
        if self.queue:
            return self.queue.popleft()
        return None
    
    def mark_sent(self, message_id: str):
        """Mark message as sent (for deduplication)."""
        self.sent_ids.add(message_id)
        self.sent_count += 1
        
        # Cleanup old sent IDs (simple LRU)
        if len(self.sent_ids) > 1000:
            # Remove oldest 200
            to_remove = list(self.sent_ids)[:200]
            for msg_id in to_remove:
                self.sent_ids.discard(msg_id)
    
    def mark_failed(self):
        """Mark message as failed."""
        self.failed_count += 1
    
    def size(self) -> int:
        """Get queue size."""
        return len(self.queue)
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self.queue) == 0


class SMSClient:
    """
    Production SMS client with background sending and retry logic.
    Never blocks calls.
    """
    
    def __init__(self):
        self.client: Optional[Client] = None
        self.queue = SMSQueue()
        self.from_number: Optional[str] = None
        
        # Background tasks
        self.processor_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Initialize Twilio client
        self._initialize_client()
        
        logger.info("SMSClient initialized")
    
    def _initialize_client(self):
        """Initialize Twilio client."""
        try:
            account_sid = os.getenv("TWILIO_ACCOUNT_SID")
            auth_token = os.getenv("TWILIO_AUTH_TOKEN")
            self.from_number = os.getenv("TWILIO_PHONE_NUMBER")
            
            if not account_sid or not auth_token or not self.from_number:
                logger.error(
                    "TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and "
                    "TWILIO_PHONE_NUMBER required"
                )
                return
            
            self.client = Client(account_sid, auth_token)
            logger.info(f"Twilio client initialized (from: {self.from_number})")
        
        except Exception as e:
            logger.error(f"Failed to initialize Twilio client: {str(e)}")
    
    async def start(self):
        """Start background SMS processor."""
        if self.is_running:
            return
        
        self.is_running = True
        self.processor_task = asyncio.create_task(
            self._processor_loop()
        )
        logger.info("SMS processor started")
    
    async def stop(self):
        """Stop background SMS processor."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.processor_task and not self.processor_task.done():
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining messages
        await self._flush_queue()
        
        logger.info("SMS processor stopped")
    
    async def _processor_loop(self):
        """Background loop to process SMS queue."""
        try:
            while self.is_running:
                await asyncio.sleep(SMS_PROCESSING_INTERVAL)
                
                if self.queue.is_empty():
                    continue
                
                await self._process_next_message()
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"SMS processor error: {str(e)}")
    
    async def _process_next_message(self):
        """Process next message in queue."""
        message = self.queue.dequeue()
        
        if not message:
            return
        
        success = await self._send_with_retry(message)
        
        if success:
            self.queue.mark_sent(message.message_id)
        else:
            self.queue.mark_failed()
            await self._log_failure(message)
    
    async def _send_with_retry(self, message: SMSMessage) -> bool:
        """
        Send SMS with retry logic.
        
        Args:
            message: SMS message
            
        Returns:
            True if sent successfully
        """
        if not self.client or not self.from_number:
            logger.error("SMS client not initialized")
            return False
        
        for attempt in range(MAX_RETRIES + 1):
            message.attempts = attempt + 1
            message.last_attempt = datetime.utcnow()
            
            try:
                # Send in executor (blocking Twilio API)
                loop = asyncio.get_event_loop()
                
                twilio_message = await loop.run_in_executor(
                    None,
                    lambda: self.client.messages.create(
                        body=message.message,
                        from_=self.from_number,
                        to=message.to_number
                    )
                )
                
                logger.info(
                    f"SMS sent successfully: {message.message_id} "
                    f"(SID: {twilio_message.sid})"
                )
                return True
            
            except TwilioRestException as e:
                logger.error(
                    f"Twilio error (attempt {attempt + 1}): "
                    f"{e.code} - {e.msg}"
                )
                message.error = f"{e.code}: {e.msg}"
                
                # Don't retry certain errors
                if e.code in [21211, 21614]:  # Invalid number
                    logger.error(f"Invalid number, not retrying: {message.to_number}")
                    return False
                
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            
            except Exception as e:
                logger.error(f"SMS send error (attempt {attempt + 1}): {str(e)}")
                message.error = str(e)
                
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
        
        logger.error(f"SMS failed after {MAX_RETRIES + 1} attempts: {message.message_id}")
        return False
    
    async def _flush_queue(self):
        """Flush all pending messages."""
        logger.info(f"Flushing {self.queue.size()} pending SMS messages")
        
        while not self.queue.is_empty():
            await self._process_next_message()
    
    async def _log_failure(self, message: SMSMessage):
        """Log failed SMS to database."""
        try:
            from db import db
            
            db.store_call_log({
                "restaurant_id": "system",
                "call_sid": f"sms_{message.message_id}",
                "direction": "outbound",
                "status": "failed",
                "transcript": f"Failed SMS: {message.to_dict()}"
            })
        except Exception as e:
            logger.error(f"Failed to log SMS failure: {str(e)}")
    
    # ========================================================================
    # PUBLIC API (Non-blocking)
    # ========================================================================
    
    def send_message(
        self,
        to_number: str,
        message: str,
        message_id: Optional[str] = None
    ) -> bool:
        """
        Send SMS message (non-blocking).
        
        Args:
            to_number: Recipient phone number (E.164 format)
            message: Message text
            message_id: Optional message ID for idempotency
            
        Returns:
            True if enqueued successfully
            
        Example:
            >>> sms_client.send_message("+1234567890", "Your order is ready!")
        """
        if not to_number or not message:
            logger.warning("Phone number and message required")
            return False
        
        # Clean phone number
        to_number = self._clean_phone_number(to_number)
        
        # Validate phone number format
        if not self._is_valid_phone(to_number):
            logger.warning(f"Invalid phone number format: {to_number}")
            return False
        
        # Create message
        sms = SMSMessage(to_number, message, message_id)
        
        # Enqueue
        return self.queue.enqueue(sms)
    
    def _clean_phone_number(self, phone: str) -> str:
        """Clean phone number to E.164 format."""
        # Remove non-digits
        digits = ''.join(c for c in phone if c.isdigit())
        
        # Add + prefix if missing
        if not phone.startswith('+'):
            if len(digits) == 10:
                # US number
                return f"+1{digits}"
            elif len(digits) == 11 and digits.startswith('1'):
                return f"+{digits}"
            else:
                return f"+{digits}"
        
        return phone
    
    def _is_valid_phone(self, phone: str) -> bool:
        """Validate phone number format (basic check)."""
        if not phone.startswith('+'):
            return False
        
        digits = phone[1:]
        if not digits.isdigit():
            return False
        
        if len(digits) < 10 or len(digits) > 15:
            return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get SMS statistics."""
        return {
            "queue_size": self.queue.size(),
            "sent_count": self.queue.sent_count,
            "failed_count": self.queue.failed_count,
            "dropped_count": self.queue.dropped_count,
            "is_running": self.is_running
        }
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.queue.size()
    
    def is_healthy(self) -> bool:
        """Check if SMS client is healthy."""
        return self.client is not None and self.is_running


# ============================================================================
# GLOBAL SMS INSTANCE
# ============================================================================

sms_client = SMSClient()


# ============================================================================
# AUTO-START PROCESSOR
# ============================================================================

async def _auto_start():
    """Auto-start SMS processor."""
    await sms_client.start()


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
# CONVENIENCE FUNCTIONS
# ============================================================================

def send_order_confirmation(
    phone: str,
    order: Dict[str, Any]
) -> bool:
    """
    Send order confirmation SMS.
    
    Args:
        phone: Customer phone number
        order: Order data
        
    Returns:
        True if enqueued
    """
    order_id = order.get("order_id", "N/A")
    total = order.get("total", 0.0)
    restaurant_name = "Captain Jay's Fish & Chicken"
    
    message = (
        f"Order confirmed! {restaurant_name}\n"
        f"Order #{order_id}\n"
        f"Total: ${total:.2f}\n"
        f"Thank you for your order!"
    )
    
    # Use order_id for idempotency
    message_id = f"order_{order_id}"
    
    return sms_client.send_message(phone, message, message_id)


def send_custom_message(phone: str, message: str) -> bool:
    """
    Send custom SMS message.
    
    Args:
        phone: Recipient phone number
        message: Message text
        
    Returns:
        True if enqueued
    """
    return sms_client.send_message(phone, message)


# ============================================================================
# GRACEFUL SHUTDOWN
# ============================================================================

async def shutdown_sms():
    """Graceful SMS shutdown."""
    await sms_client.stop()
    logger.info("SMS shutdown complete")


# ============================================================================
# MONITORING
# ============================================================================

def get_sms_stats() -> Dict[str, Any]:
    """Get SMS statistics."""
    return sms_client.get_stats()


def is_sms_healthy() -> bool:
    """Check SMS health."""
    return sms_client.is_healthy()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def example():
        print("SMS Module (Production)")
        print("=" * 50)
        
        # Start processor
        await sms_client.start()
        print("\nSMS processor started")
        
        # Send messages (non-blocking)
        print("\nSending messages...")
        
        sms_client.send_message(
            "+1234567890",
            "Your order is ready for pickup!"
        )
        
        order = {
            "order_id": "ord_123",
            "total": 25.99
        }
        send_order_confirmation("+1234567890", order)
        
        print(f"Queue size: {sms_client.get_queue_size()}")
        
        # Stats
        print("\nSMS stats:")
        stats = get_sms_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Health check
        print(f"\nHealthy: {is_sms_healthy()}")
        
        # Wait for processing
        await asyncio.sleep(3)
        
        # Shutdown
        await shutdown_sms()
        print("\nSMS shutdown complete")
        
        print("\n" + "=" * 50)
        print("Production SMS module ready")
    
    asyncio.run(example())
