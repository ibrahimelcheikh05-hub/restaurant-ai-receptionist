"""
Cancellation Token System (Production)
=======================================
Concurrent cancellation for STT, LLM, and TTS operations.

Features:
- Thread-safe cancellation
- Hierarchical token scopes
- Operation-specific tokens
- Timeout integration
"""

import asyncio
import logging
from typing import Optional, Set
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of cancellable operations."""
    STT = "stt"
    LLM = "llm"
    TTS = "tts"
    ALL = "all"


class CancelToken:
    """
    Cancellation token for async operations.
    
    Thread-safe cancellation primitive that can be checked
    and awaited by concurrent operations.
    """
    
    def __init__(self, call_id: str, operation_type: OperationType = OperationType.ALL):
        self.call_id = call_id
        self.operation_type = operation_type
        self._cancelled = False
        self._cancel_event = asyncio.Event()
        self._cancel_reason: Optional[str] = None
        self._cancelled_at: Optional[datetime] = None
        self._lock = asyncio.Lock()
        
    @property
    def is_cancelled(self) -> bool:
        """Check if token is cancelled (non-blocking)."""
        return self._cancelled
    
    @property
    def cancel_reason(self) -> Optional[str]:
        """Get cancellation reason."""
        return self._cancel_reason
    
    async def cancel(self, reason: str = "manual"):
        """
        Cancel the token.
        
        Args:
            reason: Reason for cancellation
        """
        async with self._lock:
            if self._cancelled:
                return  # Already cancelled
            
            self._cancelled = True
            self._cancel_reason = reason
            self._cancelled_at = datetime.utcnow()
            self._cancel_event.set()
        
        logger.info(
            f"Token cancelled: {self.operation_type.value}",
            extra={
                "call_id": self.call_id,
                "operation_type": self.operation_type.value,
                "reason": reason
            }
        )
    
    async def wait_cancelled(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for cancellation with optional timeout.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            True if cancelled, False if timeout
        """
        try:
            if timeout:
                await asyncio.wait_for(self._cancel_event.wait(), timeout=timeout)
            else:
                await self._cancel_event.wait()
            return True
        except asyncio.TimeoutError:
            return False
    
    def check_cancelled(self):
        """
        Check if cancelled and raise if so.
        
        Raises:
            asyncio.CancelledError: If token is cancelled
        """
        if self._cancelled:
            raise asyncio.CancelledError(
                f"Operation cancelled: {self._cancel_reason}"
            )
    
    def get_status(self) -> dict:
        """Get token status."""
        return {
            "call_id": self.call_id,
            "operation_type": self.operation_type.value,
            "is_cancelled": self._cancelled,
            "cancel_reason": self._cancel_reason,
            "cancelled_at": self._cancelled_at.isoformat() if self._cancelled_at else None
        }


class CancelTokenRegistry:
    """
    Registry for managing multiple cancellation tokens per call.
    
    Allows cancelling specific operations or all operations for a call.
    """
    
    def __init__(self):
        self._tokens: dict[str, dict[OperationType, CancelToken]] = {}
        self._lock = asyncio.Lock()
    
    async def create_token(
        self,
        call_id: str,
        operation_type: OperationType
    ) -> CancelToken:
        """
        Create a new cancellation token.
        
        Args:
            call_id: Call identifier
            operation_type: Type of operation
            
        Returns:
            New CancelToken
        """
        async with self._lock:
            if call_id not in self._tokens:
                self._tokens[call_id] = {}
            
            token = CancelToken(call_id, operation_type)
            self._tokens[call_id][operation_type] = token
            
            logger.debug(
                f"Token created: {operation_type.value}",
                extra={"call_id": call_id, "operation_type": operation_type.value}
            )
            
            return token
    
    async def get_token(
        self,
        call_id: str,
        operation_type: OperationType
    ) -> Optional[CancelToken]:
        """Get existing token."""
        async with self._lock:
            return self._tokens.get(call_id, {}).get(operation_type)
    
    async def cancel_operation(
        self,
        call_id: str,
        operation_type: OperationType,
        reason: str = "manual"
    ):
        """
        Cancel specific operation type for a call.
        
        Args:
            call_id: Call identifier
            operation_type: Type of operation to cancel
            reason: Cancellation reason
        """
        async with self._lock:
            if call_id not in self._tokens:
                return
            
            token = self._tokens[call_id].get(operation_type)
            if token:
                await token.cancel(reason)
    
    async def cancel_all(self, call_id: str, reason: str = "call_ended"):
        """
        Cancel all operations for a call.
        
        Args:
            call_id: Call identifier
            reason: Cancellation reason
        """
        async with self._lock:
            if call_id not in self._tokens:
                return
            
            tokens = self._tokens[call_id].values()
        
        # Cancel outside lock to avoid deadlock
        for token in tokens:
            await token.cancel(reason)
        
        logger.info(
            f"All tokens cancelled for call",
            extra={"call_id": call_id, "reason": reason, "token_count": len(tokens)}
        )
    
    async def cleanup(self, call_id: str):
        """
        Clean up all tokens for a call.
        
        Args:
            call_id: Call identifier
        """
        async with self._lock:
            if call_id in self._tokens:
                count = len(self._tokens[call_id])
                del self._tokens[call_id]
                
                logger.debug(
                    f"Tokens cleaned up",
                    extra={"call_id": call_id, "token_count": count}
                )
    
    async def get_all_tokens(self, call_id: str) -> dict[OperationType, CancelToken]:
        """Get all tokens for a call."""
        async with self._lock:
            return dict(self._tokens.get(call_id, {}))
    
    def get_status(self) -> dict:
        """Get registry status."""
        return {
            "total_calls": len(self._tokens),
            "total_tokens": sum(len(tokens) for tokens in self._tokens.values())
        }


# Global registry instance
_token_registry = CancelTokenRegistry()


async def create_cancel_token(
    call_id: str,
    operation_type: OperationType
) -> CancelToken:
    """Create a new cancellation token."""
    return await _token_registry.create_token(call_id, operation_type)


async def get_cancel_token(
    call_id: str,
    operation_type: OperationType
) -> Optional[CancelToken]:
    """Get existing cancellation token."""
    return await _token_registry.get_token(call_id, operation_type)


async def cancel_operation(
    call_id: str,
    operation_type: OperationType,
    reason: str = "manual"
):
    """Cancel specific operation."""
    await _token_registry.cancel_operation(call_id, operation_type, reason)


async def cancel_all_operations(call_id: str, reason: str = "call_ended"):
    """Cancel all operations for a call."""
    await _token_registry.cancel_all(call_id, reason)


async def cleanup_tokens(call_id: str):
    """Clean up tokens for a call."""
    await _token_registry.cleanup(call_id)
