"""
Event Handlers (Business Logic Layer)
======================================
External event handlers called by CallController.

This layer:
- Contains business logic
- Accesses databases
- Calls AI services
- Manages domain entities (orders, menus, etc.)

Separated from orchestration to enable:
- Clean testing
- Easy business logic changes
- Clear separation of concerns
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import json

# Import existing modules (unchanged)
from memory import create_call_memory, get_memory, clear_memory
from db import db
from menu import get_menu, format_menu_for_prompt
from order import (
    create_order,
    get_order_summary,
    validate_order,
    finalize_order
)
from upsell import suggest_upsells, format_suggestion_text
from detect import detect_language
from translate import to_english, from_english
from cancel_token import CancelToken

# AI client
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# OpenAI client (singleton)
_openai_client = None


def _get_openai_client() -> AsyncOpenAI:
    """Get or create OpenAI client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


async def handle_greeting(
    call_id: str,
    restaurant_id: str,
    customer_phone: Optional[str] = None
) -> Dict[str, Any]:
    """
    Handle call greeting.
    
    Args:
        call_id: Call identifier
        restaurant_id: Restaurant ID
        customer_phone: Customer phone number
        
    Returns:
        Greeting response
    """
    logger.info(f"Handling greeting for call {call_id}")
    
    try:
        # Create memory
        memory = create_call_memory(call_id, restaurant_id)
        
        # Load menu
        menu = await get_menu(restaurant_id)
        memory.set_menu_snapshot(menu)
        
        # Create order
        create_order(call_id, restaurant_id)
        
        # Log to database
        db.store_call_log({
            "restaurant_id": restaurant_id,
            "caller_phone": customer_phone or "unknown",
            "call_sid": call_id,
            "direction": "inbound",
            "status": "in-progress",
            "transcript": "Call started"
        })
        
        # Get greeting text
        default_language = os.getenv("DEFAULT_LANGUAGE", "en")
        greeting = "Thank you for calling Captain Jay's Fish & Chicken! How can I help you today?"
        
        return {
            "greeting": greeting,
            "language": default_language
        }
    
    except Exception as e:
        logger.error(f"Greeting handler error: {str(e)}", exc_info=True)
        return {
            "greeting": "Welcome! How can I help you?",
            "language": "en"
        }


async def handle_ai_request(
    call_id: str,
    user_text: str,
    language: Optional[str] = None,
    cancel_token: Optional[CancelToken] = None
) -> Dict[str, Any]:
    """
    Handle AI request with business context.
    
    Args:
        call_id: Call identifier
        user_text: User input text
        language: Language code
        cancel_token: Cancellation token
        
    Returns:
        AI response
    """
    logger.info(f"Handling AI request for call {call_id}")
    
    try:
        # Get memory
        memory = get_memory(call_id)
        if not memory:
            logger.error(f"Memory not found for {call_id}")
            return await handle_ai_fallback(call_id, "memory_error")
        
        # Translate to English if needed
        if language and language != "en":
            user_text = await to_english(user_text, language)
        
        # Add to conversation history
        memory.add_conversation_turn(
            role="user",
            content=user_text,
            intent="order_inquiry"
        )
        
        # Build AI context
        menu_text = format_menu_for_prompt(memory.get_menu_snapshot())
        order_summary = get_order_summary(call_id)
        conversation_history = memory.get_conversation_history()
        
        # Get upsell suggestions
        upsell_suggestions = suggest_upsells(call_id, memory.get_menu_snapshot())
        upsell_text = format_suggestion_text(upsell_suggestions) if upsell_suggestions else ""
        
        # Build prompt
        system_prompt = os.getenv(
            "SYSTEM_AGENT_PROMPT",
            "You are a professional restaurant phone order assistant."
        )
        
        context = f"""
Current Menu:
{menu_text}

Current Order:
{json.dumps(order_summary, indent=2) if order_summary else "No items yet"}

{f"Suggested Upsells: {upsell_text}" if upsell_text else ""}

Conversation History:
{json.dumps(conversation_history[-5:], indent=2)}
"""
        
        user_message = f"{context}\n\nCustomer: {user_text}\n\nRespond naturally and helpfully:"
        
        # Call OpenAI with cancellation support
        client = _get_openai_client()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        
        # Create task that can be cancelled
        async def _call_openai():
            response = await client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages
            )
            return response.choices[0].message.content
        
        # Run with cancellation check
        ai_task = asyncio.create_task(_call_openai())
        
        # Periodically check cancellation
        while not ai_task.done():
            if cancel_token and cancel_token.is_cancelled:
                ai_task.cancel()
                logger.info(f"AI request cancelled for {call_id}")
                raise asyncio.CancelledError("AI request cancelled")
            
            await asyncio.sleep(0.1)
        
        ai_text = await ai_task
        
        # Add to memory
        memory.add_conversation_turn(
            role="assistant",
            content=ai_text,
            intent="response"
        )
        
        # Translate back if needed
        response_text = ai_text
        if language and language != "en":
            response_text = await from_english(ai_text, language)
        
        # Detect if end call is suggested
        suggested_action = None
        if any(phrase in ai_text.lower() for phrase in ["goodbye", "thank you for calling"]):
            suggested_action = "end_call"
        
        return {
            "response_text": response_text,
            "language": language or "en",
            "suggested_action": suggested_action
        }
    
    except asyncio.CancelledError:
        logger.info(f"AI request cancelled for {call_id}")
        raise
    
    except Exception as e:
        logger.error(f"AI request error: {str(e)}", exc_info=True)
        return await handle_ai_fallback(call_id, "exception")


async def handle_ai_fallback(
    call_id: str,
    error_type: str
) -> Dict[str, Any]:
    """
    Handle AI fallback response.
    
    Args:
        call_id: Call identifier
        error_type: Type of error
        
    Returns:
        Fallback response
    """
    logger.warning(f"AI fallback for call {call_id}: {error_type}")
    
    fallback_responses = {
        "timeout": "I apologize, could you please repeat that?",
        "memory_error": "I'm having trouble. Could you start over?",
        "state_error": "Let me help you. What would you like to order?",
        "exception": "I apologize for the confusion. How can I help you?",
    }
    
    response_text = fallback_responses.get(
        error_type,
        "I'm here to help. What would you like to order?"
    )
    
    return {
        "response_text": response_text,
        "language": "en",
        "suggested_action": None
    }


async def handle_call_end(
    call_id: str,
    restaurant_id: str,
    customer_phone: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Handle call end and cleanup.
    
    Args:
        call_id: Call identifier
        restaurant_id: Restaurant ID
        customer_phone: Customer phone number
        stats: Call statistics
        
    Returns:
        Cleanup summary
    """
    logger.info(f"Handling call end for {call_id}")
    
    try:
        # Get memory
        memory = get_memory(call_id)
        
        # Finalize order if valid
        order_finalized = False
        if memory:
            try:
                order_summary = get_order_summary(call_id)
                
                if order_summary and validate_order(call_id):
                    final_order = finalize_order(call_id, customer_phone=customer_phone)
                    
                    if final_order:
                        logger.info(f"Order finalized: {final_order['order_id']}")
                        order_finalized = True
                        
                        # Send SMS confirmation
                        if customer_phone:
                            try:
                                from sms import send_order_confirmation
                                send_order_confirmation(customer_phone, final_order)
                            except Exception as e:
                                logger.error(f"Failed to send SMS: {str(e)}")
            
            except Exception as e:
                logger.error(f"Error finalizing order: {str(e)}")
        
        # Update call log
        duration = stats.get("duration_seconds", 0) if stats else 0
        turns = stats.get("safety_stats", {}).get("turn_count", 0) if stats else 0
        
        db.store_call_log({
            "restaurant_id": restaurant_id,
            "caller_phone": customer_phone or "unknown",
            "call_sid": call_id,
            "direction": "inbound",
            "status": "completed",
            "duration": int(duration),
            "transcript": f"Call ended. Duration: {int(duration)}s, Turns: {turns}"
        })
        
        # Clear memory
        clear_memory(call_id)
        
        return {
            "order_finalized": order_finalized,
            "duration_seconds": duration,
            "turns": turns
        }
    
    except Exception as e:
        logger.error(f"Call end handler error: {str(e)}", exc_info=True)
        return {
            "order_finalized": False,
            "error": str(e)
        }


async def handle_transfer_request(
    call_id: str,
    reason: str
) -> Dict[str, Any]:
    """
    Handle call transfer request.
    
    Args:
        call_id: Call identifier
        reason: Transfer reason
        
    Returns:
        Transfer details
    """
    logger.info(f"Handling transfer request for {call_id}: {reason}")
    
    try:
        from config import get_config, is_feature_enabled
        
        if not is_feature_enabled("call_transfer"):
            return {
                "transfer_requested": False,
                "reason": "Transfer feature disabled"
            }
        
        config = get_config()
        transfer_number = config.twilio.human_transfer_number
        
        if not transfer_number:
            raise RuntimeError("Transfer number not configured")
        
        # Update memory
        memory = get_memory(call_id)
        if memory:
            memory.set_state("transfer_requested")
            memory.add_conversation_turn(
                role="system",
                content=f"Transfer requested: {reason}",
                intent="transfer"
            )
        
        return {
            "transfer_requested": True,
            "transfer_number": transfer_number,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Transfer request error: {str(e)}")
        return {
            "transfer_requested": False,
            "error": str(e)
        }
