"""
Main AI Orchestrator
====================
Central brain of the voice ordering system.
Coordinates all modules and manages conversation flow.
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

# Import all modules
from memory import create_call_memory, get_memory, clear_memory, CallMemory
from db import db
from menu import get_menu, format_menu_for_prompt
from order import (
    create_order, add_item, remove_item, update_item_quantity,
    calculate_total, finalize_order, get_order_summary, validate_order
)
from upsell import suggest_upsells, format_suggestion_text
from detect import detect_language
from translate import to_english, from_english


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# AI Provider Configuration
AI_PROVIDER = os.getenv("AI_PROVIDER", "claude")  # "claude" or "openai"

# Claude Configuration
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

# System Prompt (loaded from environment)
SYSTEM_AGENT_PROMPT = os.getenv(
    "SYSTEM_AGENT_PROMPT",
    """You are a professional restaurant phone order assistant.

Your responsibilities:
1. Greet customers warmly
2. Help them browse the menu
3. Take accurate orders
4. Suggest complementary items (upsells)
5. Confirm order details
6. Collect delivery/pickup information
7. Provide order total

Guidelines:
- Be friendly and professional
- Listen carefully to customer requests
- Clarify any ambiguities
- Suggest items when appropriate (but don't be pushy)
- Confirm all details before finalizing
- Keep responses concise and natural

Order Process:
1. Greet customer
2. Take order (one or more items)
3. Suggest complementary items if appropriate
4. Confirm order details
5. Get customer information (name, phone, address if delivery)
6. Provide total and estimated time
7. Thank customer and end call
"""
)

# Conversation settings
MAX_CONVERSATION_TURNS = 50
DEFAULT_LANGUAGE = "en"


# ============================================================================
# AI CLIENT INITIALIZATION
# ============================================================================

def _get_ai_client():
    """Get AI client based on provider configuration."""
    if AI_PROVIDER.lower() == "claude":
        if not CLAUDE_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable not set for Claude"
            )
        return AsyncAnthropic(api_key=CLAUDE_API_KEY)
    
    elif AI_PROVIDER.lower() == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable not set for OpenAI"
            )
        return AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    else:
        raise ValueError(
            f"Unknown AI_PROVIDER: {AI_PROVIDER}. Use 'claude' or 'openai'"
        )


# Global AI client (lazy-loaded)
_ai_client = None


def get_ai_client():
    """Get or create AI client."""
    global _ai_client
    if _ai_client is None:
        _ai_client = _get_ai_client()
    return _ai_client


# ============================================================================
# CALL LIFECYCLE HANDLERS
# ============================================================================

async def handle_call_start(
    call_id: str,
    restaurant_id: str,
    customer_phone: Optional[str] = None
) -> Dict[str, Any]:
    """
    Initialize a new call session.
    
    Args:
        call_id: Unique call identifier
        restaurant_id: Restaurant identifier
        customer_phone: Optional customer phone number
        
    Returns:
        Call initialization status
        
    Raises:
        RuntimeError: If initialization fails
        
    Example:
        >>> result = await handle_call_start("CA123", "rest_123")
    """
    try:
        # 1. Create memory for this call
        memory = create_call_memory(call_id)
        memory.set_restaurant_id(restaurant_id)
        memory.set_state("call_active")
        
        if customer_phone:
            memory.set_customer_info(phone=customer_phone)
        
        # 2. Load menu
        menu = get_menu(restaurant_id)
        memory.set_menu_snapshot(menu.get("items", []))
        
        # 3. Create order
        create_order(call_id, restaurant_id)
        
        # 4. Initialize call log in database
        call_log_data = {
            "restaurant_id": restaurant_id,
            "caller_phone": customer_phone or "unknown",
            "call_sid": call_id,
            "direction": "inbound",
            "status": "in-progress"
        }
        call_log = db.store_call_log(call_log_data)
        
        return {
            "status": "initialized",
            "call_id": call_id,
            "restaurant_id": restaurant_id,
            "menu_items": len(menu.get("items", [])),
            "call_log_id": call_log.get("id"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        # Clean up on failure
        clear_memory(call_id)
        raise RuntimeError(f"Failed to initialize call: {str(e)}")


async def handle_user_text(
    text: str,
    call_id: str,
    detected_language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process user input text through the full AI pipeline.
    
    This is the main conversation handler that:
    1. Detects/validates language
    2. Translates to English if needed
    3. Processes through AI
    4. Translates response back to user language
    5. Returns response for TTS
    
    Args:
        text: User input text (transcribed from speech)
        call_id: Call identifier
        detected_language: Optional pre-detected language
        
    Returns:
        Dictionary with:
        - response_text: AI response in user's language
        - language: User's language
        - intent: Detected intent (if any)
        - actions: Any actions taken (order updates, etc.)
        
    Raises:
        RuntimeError: If processing fails
        
    Example:
        >>> result = await handle_user_text("I want pizza", "CA123")
        >>> print(result["response_text"])
    """
    try:
        # Get memory
        memory = get_memory(call_id)
        if not memory:
            raise RuntimeError(
                f"No active call found for call_id '{call_id}'. "
                f"Call handle_call_start() first."
            )
        
        # 1. LANGUAGE DETECTION
        if not detected_language:
            detection = detect_language(text)
            user_language = detection["language"]
            
            # Update memory with detected language
            if not memory.detected_language:
                memory.update_language(user_language)
        else:
            user_language = detected_language
        
        # Store user language in memory
        if not memory.detected_language:
            memory.update_language(user_language)
        
        # 2. TRANSLATION TO ENGLISH (if needed)
        if user_language != "en":
            english_text = await to_english(text, user_language)
        else:
            english_text = text
        
        # 3. ADD TO CONVERSATION CONTEXT
        memory.add_conversation_turn(
            role="user",
            content=english_text,
            intent=None  # Will be determined by AI
        )
        
        # 4. BUILD AI PROMPT
        prompt = await _build_ai_prompt(call_id, english_text, memory)
        
        # 5. CALL AI MODEL
        ai_response = await _call_ai_model(prompt, memory)
        
        # 6. PROCESS AI RESPONSE
        actions_taken = await _process_ai_response(
            ai_response,
            call_id,
            memory
        )
        
        # 6.5. CHECK FOR TRANSFER INTENT
        transfer_detected = detect_transfer_intent(english_text, ai_response)
        
        if transfer_detected:
            logger.info(f"Transfer intent detected for call {call_id}")
            
            # Request transfer
            transfer_result = await request_call_transfer(
                call_id,
                reason="Customer requested to speak with human agent"
            )
            
            # Add transfer flag to actions
            actions_taken["transfer_requested"] = True
            actions_taken["transfer_details"] = transfer_result
        
        # 7. ADD AI RESPONSE TO CONTEXT
        memory.add_conversation_turn(
            role="assistant",
            content=ai_response,
            intent=actions_taken.get("intent")
        )
        
        # 8. TRANSLATE RESPONSE BACK TO USER LANGUAGE
        if user_language != "en":
            response_text = await from_english(ai_response, user_language)
        else:
            response_text = ai_response
        
        return {
            "response_text": response_text,
            "language": user_language,
            "intent": actions_taken.get("intent"),
            "actions": actions_taken,
            "original_text": text,
            "english_text": english_text
        }
        
    except Exception as e:
        # Log error but don't crash
        error_msg = f"Error processing user text: {str(e)}"
        print(f"ERROR: {error_msg}")
        
        # Return fallback response
        fallback = "I apologize, I'm having trouble processing that. Could you please repeat?"
        
        # Translate fallback if needed
        if user_language and user_language != "en":
            try:
                fallback = await from_english(fallback, user_language)
            except:
                pass  # Use English fallback if translation fails
        
        return {
            "response_text": fallback,
            "language": user_language or DEFAULT_LANGUAGE,
            "intent": "error",
            "actions": {"error": error_msg},
            "error": True
        }


async def handle_call_end(call_id: str) -> Dict[str, Any]:
    """
    Finalize call and clean up resources.
    
    Args:
        call_id: Call identifier
        
    Returns:
        Call summary and cleanup status
        
    Example:
        >>> summary = await handle_call_end("CA123")
    """
    try:
        # Get memory
        memory = get_memory(call_id)
        
        if not memory:
            return {
                "status": "no_active_call",
                "call_id": call_id
            }
        
        # Get final state
        order_summary = get_order_summary(call_id)
        conversation = memory.get_recent_context(num_turns=100)
        
        # Update call state
        memory.set_state("call_completed")
        
        # Try to finalize order if it has items
        order_finalized = False
        order_id = None
        
        if not order_summary["is_empty"]:
            try:
                # Validate and finalize order
                is_valid, errors = validate_order(call_id)
                
                if is_valid:
                    final_order = finalize_order(
                        call_id,
                        customer_phone=memory.customer_phone,
                        customer_name=memory.customer_name
                    )
                    order_finalized = True
                    order_id = final_order.get("order_id")
            except Exception as e:
                print(f"Warning: Could not finalize order: {str(e)}")
        
        # Update call log in database
        try:
            call_log_updates = {
                "status": "completed",
                "duration": int((datetime.utcnow() - memory.call_start_time).total_seconds()),
                "transcript": "\n".join([
                    f"{turn['role']}: {turn['content']}"
                    for turn in conversation
                ])
            }
            
            # Find call log by call_sid
            # Note: This assumes we stored call_log_id in memory or can find by call_sid
            # For now, we'll use a simple approach
            
        except Exception as e:
            print(f"Warning: Could not update call log: {str(e)}")
        
        # Get summary
        summary = {
            "status": "completed",
            "call_id": call_id,
            "restaurant_id": memory.restaurant_id,
            "order_finalized": order_finalized,
            "order_id": order_id,
            "total_turns": memory.turn_count,
            "order_summary": order_summary,
            "customer_phone": memory.customer_phone,
            "customer_name": memory.customer_name,
            "detected_language": memory.detected_language,
            "call_duration_seconds": int(
                (datetime.utcnow() - memory.call_start_time).total_seconds()
            )
        }
        
        # Clear memory
        clear_memory(call_id)
        
        return summary
        
    except Exception as e:
        # Clean up memory even on error
        clear_memory(call_id)
        raise RuntimeError(f"Failed to end call: {str(e)}")


# ============================================================================
# AI PROMPT CONSTRUCTION
# ============================================================================

async def _build_ai_prompt(
    call_id: str,
    user_text: str,
    memory: CallMemory
) -> str:
    """
    Build comprehensive AI prompt with context.
    
    Args:
        call_id: Call identifier
        user_text: User's message in English
        memory: Call memory instance
        
    Returns:
        Complete prompt for AI
    """
    # Get current order
    order_summary = get_order_summary(call_id)
    
    # Get menu
    menu = {
        "items": memory.menu_snapshot,
        "has_items": len(memory.menu_snapshot) > 0
    }
    formatted_menu = format_menu_for_prompt(menu)
    
    # Get upsell suggestions
    current_order_items = order_summary["items"]
    suggestions = suggest_upsells(menu, current_order_items, max_suggestions=3)
    
    upsell_text = ""
    if suggestions:
        upsell_text = "\n\nSUGGESTED UPSELLS (mention naturally if appropriate):\n"
        for suggestion in suggestions:
            upsell_text += f"- {format_suggestion_text(suggestion)}\n"
    
    # Get conversation history
    recent_context = memory.get_recent_context(num_turns=10)
    conversation_history = ""
    if recent_context:
        conversation_history = "\n\nRECENT CONVERSATION:\n"
        for turn in recent_context[:-1]:  # Exclude current turn
            conversation_history += f"{turn['role'].upper()}: {turn['content']}\n"
    
    # Build current order text
    order_text = "\n\nCURRENT ORDER:\n"
    if order_summary["is_empty"]:
        order_text += "Empty (no items yet)\n"
    else:
        for item in order_summary["items"]:
            customizations = ", ".join(item.get("customizations", []))
            custom_text = f" ({customizations})" if customizations else ""
            order_text += f"- {item['quantity']}x {item['name']}{custom_text} @ ${item['price']}\n"
        order_text += f"\nSubtotal: ${order_summary['subtotal']:.2f}\n"
        order_text += f"Tax: ${order_summary['tax']:.2f}\n"
        order_text += f"TOTAL: ${order_summary['total']:.2f}\n"
    
    # Customer info
    customer_info = "\n\nCUSTOMER INFO:\n"
    if memory.customer_phone:
        customer_info += f"Phone: {memory.customer_phone}\n"
    if memory.customer_name:
        customer_info += f"Name: {memory.customer_name}\n"
    if not memory.customer_phone and not memory.customer_name:
        customer_info += "Not yet collected\n"
    
    # Build complete prompt
    prompt = f"""SYSTEM: {SYSTEM_AGENT_PROMPT}

{formatted_menu}

{order_text}

{customer_info}

{upsell_text}

{conversation_history}

USER: {user_text}

ASSISTANT: """
    
    return prompt


# ============================================================================
# AI MODEL INTERACTION
# ============================================================================

async def _call_ai_model(
    prompt: str,
    memory: CallMemory
) -> str:
    """
    Call AI model (Claude or OpenAI) with prompt.
    
    Args:
        prompt: Complete prompt
        memory: Call memory
        
    Returns:
        AI response text
    """
    client = get_ai_client()
    
    try:
        if AI_PROVIDER.lower() == "claude":
            # Call Claude API
            response = await client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                temperature=0.7,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Extract text from response
            return response.content[0].text
            
        elif AI_PROVIDER.lower() == "openai":
            # Call OpenAI API
            response = await client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=1024,
                temperature=0.7,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_AGENT_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            return response.choices[0].message.content
            
        else:
            raise ValueError(f"Unknown AI provider: {AI_PROVIDER}")
            
    except Exception as e:
        raise RuntimeError(f"AI model call failed: {str(e)}")


# ============================================================================
# AI RESPONSE PROCESSING
# ============================================================================

async def _process_ai_response(
    ai_response: str,
    call_id: str,
    memory: CallMemory
) -> Dict[str, Any]:
    """
    Process AI response and execute any actions.
    
    This includes:
    - Intent detection
    - Order modifications
    - Information extraction
    
    Args:
        ai_response: Response from AI
        call_id: Call identifier
        memory: Call memory
        
    Returns:
        Dictionary of actions taken
    """
    actions = {
        "intent": None,
        "order_modified": False,
        "info_collected": False
    }
    
    # Simple intent detection based on response content
    response_lower = ai_response.lower()
    
    # Detect intents
    if any(word in response_lower for word in ["added", "add", "order"]):
        actions["intent"] = "add_item"
    elif any(word in response_lower for word in ["remove", "delete", "cancel"]):
        actions["intent"] = "remove_item"
    elif any(word in response_lower for word in ["total", "confirm", "finalize"]):
        actions["intent"] = "confirm_order"
    elif any(word in response_lower for word in ["hello", "hi", "welcome"]):
        actions["intent"] = "greeting"
    else:
        actions["intent"] = "conversation"
    
    # Update last intent in memory
    if actions["intent"]:
        memory.set_last_intent(actions["intent"])
    
    return actions


# ============================================================================
# CALL TRANSFER LOGIC
# ============================================================================

def detect_transfer_intent(text: str, llm_response: str) -> bool:
    """
    Detect if user is requesting to speak to a human.
    
    Args:
        text: User's input text (in English)
        llm_response: LLM's response
        
    Returns:
        True if transfer is requested
        
    Example:
        >>> detect_transfer_intent("I want to speak to a manager", "...")
        True
    """
    # Combine user text and LLM response for comprehensive detection
    combined = f"{text.lower()} {llm_response.lower()}"
    
    # Transfer keywords and phrases
    transfer_phrases = [
        "speak to human",
        "talk to human",
        "speak to someone",
        "talk to someone",
        "speak to manager",
        "talk to manager",
        "speak to a person",
        "talk to a person",
        "transfer me",
        "connect me",
        "real person",
        "actual person",
        "live agent",
        "customer service",
        "support team",
        "representative",
        "operator",
        "staff member",
        "team member",
        "human help",
        "human support"
    ]
    
    # Check for transfer phrases in user input
    user_lower = text.lower()
    for phrase in transfer_phrases:
        if phrase in user_lower:
            logger.info(f"Transfer intent detected: '{phrase}' in user input")
            return True
    
    # Check if LLM is offering to transfer
    llm_transfer_indicators = [
        "transfer you",
        "connect you",
        "speak with a",
        "talk with a",
        "representative",
        "team member"
    ]
    
    llm_lower = llm_response.lower()
    for indicator in llm_transfer_indicators:
        if indicator in llm_lower:
            logger.info(f"Transfer offer detected in LLM response: '{indicator}'")
            return True
    
    return False


async def request_call_transfer(
    call_id: str,
    reason: str = "Customer requested human assistance"
) -> Dict[str, Any]:
    """
    Request a call transfer to human agent.
    
    This function:
    1. Updates call state to prevent further AI responses
    2. Logs transfer request
    3. Returns transfer parameters for websocket_server
    
    Args:
        call_id: Call identifier
        reason: Reason for transfer
        
    Returns:
        Dictionary with transfer details
        
    Raises:
        RuntimeError: If transfer cannot be initiated
        
    Example:
        >>> result = await request_call_transfer("CA123", "Customer request")
    """
    try:
        # Get memory
        memory = get_memory(call_id)
        if not memory:
            raise RuntimeError(f"No active call found: {call_id}")
        
        # Check if transfer is enabled
        from config import is_feature_enabled
        if not is_feature_enabled("call_transfer"):
            logger.warning(f"Transfer requested but feature disabled: {call_id}")
            return {
                "transfer_requested": False,
                "reason": "Transfer feature is disabled"
            }
        
        # Get transfer number from config
        from config import get_config
        transfer_number = get_config().twilio.human_transfer_number
        
        if not transfer_number:
            logger.error("Transfer requested but HUMAN_TRANSFER_NUMBER not configured")
            raise RuntimeError("Transfer number not configured")
        
        # Update call state
        memory.set_state("transfer_requested")
        
        # Log transfer event
        logger.info(
            f"Call transfer requested: {call_id} - Reason: {reason}"
        )
        
        # Store transfer info in memory
        memory.add_conversation_turn(
            role="system",
            content=f"Transfer requested: {reason}",
            intent="transfer"
        )
        
        # Log to database
        try:
            db.store_call_log({
                "restaurant_id": memory.restaurant_id,
                "caller_phone": memory.customer_phone or "unknown",
                "call_sid": call_id,
                "direction": "inbound",
                "status": "transferring",
                "transcript": f"Transfer requested: {reason}"
            })
        except Exception as e:
            logger.error(f"Failed to log transfer to database: {str(e)}")
        
        return {
            "transfer_requested": True,
            "call_id": call_id,
            "transfer_number": transfer_number,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to request transfer: {str(e)}", exc_info=True)
        raise RuntimeError(f"Transfer request failed: {str(e)}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_call_status(call_id: str) -> Dict[str, Any]:
    """
    Get current status of a call.
    
    Args:
        call_id: Call identifier
        
    Returns:
        Call status information
    """
    memory = get_memory(call_id)
    
    if not memory:
        return {
            "active": False,
            "call_id": call_id
        }
    
    try:
        order_summary = get_order_summary(call_id)
        
        return {
            "active": True,
            "call_id": call_id,
            "restaurant_id": memory.restaurant_id,
            "state": memory.state,
            "language": memory.detected_language,
            "turn_count": memory.turn_count,
            "order_item_count": order_summary["item_count"],
            "order_total": order_summary["total"],
            "duration_seconds": int(
                (datetime.utcnow() - memory.call_start_time).total_seconds()
            )
        }
    except:
        return {
            "active": True,
            "call_id": call_id,
            "error": "Could not retrieve full status"
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the main orchestrator.
    """
    
    async def example():
        print("Main AI Orchestrator Example")
        print("=" * 50)
        
        # Configuration
        print("\nConfiguration:")
        print(f"AI Provider: {AI_PROVIDER}")
        print(f"Model: {CLAUDE_MODEL if AI_PROVIDER == 'claude' else OPENAI_MODEL}")
        
        # Simulate a call
        call_id = "CA123example"
        restaurant_id = "rest_123"
        
        try:
            # 1. Start call
            print("\n1. Starting call...")
            start_result = await handle_call_start(call_id, restaurant_id)
            print(f"   ✓ Call initialized")
            print(f"   - Menu items: {start_result['menu_items']}")
            
            # 2. Process user input
            print("\n2. Processing user input...")
            user_texts = [
                "Hi, I'd like to order a pizza",
                "I'll have a large pepperoni pizza",
                "That's all, thank you"
            ]
            
            for i, text in enumerate(user_texts, 1):
                print(f"\n   Turn {i}:")
                print(f"   User: {text}")
                
                # Note: This would fail without proper API keys
                # result = await handle_user_text(text, call_id)
                # print(f"   Assistant: {result['response_text']}")
                
                print(f"   (Skipped - requires API key)")
            
            # 3. End call
            print("\n3. Ending call...")
            end_result = await handle_call_end(call_id)
            print(f"   ✓ Call ended")
            print(f"   - Total turns: {end_result['total_turns']}")
            print(f"   - Order finalized: {end_result['order_finalized']}")
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
            # Clean up
            clear_memory(call_id)
    
    # Run example
    asyncio.run(example())
