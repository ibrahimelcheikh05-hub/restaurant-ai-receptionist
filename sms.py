"""
SMS Notification Module
=======================
Send SMS notifications to customers using Twilio.
Handles order confirmations and other customer notifications.

NO BUSINESS LOGIC - Pure messaging service only.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

def _get_twilio_client() -> Client:
    """
    Get Twilio client instance.
    
    Returns:
        Configured Twilio client
        
    Raises:
        RuntimeError: If credentials are missing
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    
    if not account_sid or not auth_token:
        raise RuntimeError(
            "Missing Twilio credentials. "
            "Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN"
        )
    
    return Client(account_sid, auth_token)


def _get_twilio_phone() -> str:
    """
    Get Twilio phone number for sending SMS.
    
    Returns:
        Twilio phone number in E.164 format
        
    Raises:
        RuntimeError: If phone number not configured
    """
    phone = os.getenv("TWILIO_PHONE_NUMBER")
    
    if not phone:
        raise RuntimeError(
            "Missing Twilio phone number. Set TWILIO_PHONE_NUMBER"
        )
    
    if not phone.startswith("+"):
        raise RuntimeError(
            f"TWILIO_PHONE_NUMBER must be in E.164 format: {phone}"
        )
    
    return phone


# Global client (lazy-loaded)
_client: Optional[Client] = None


def _get_client() -> Client:
    """Get or create Twilio client."""
    global _client
    if _client is None:
        _client = _get_twilio_client()
    return _client


# ============================================================================
# PHONE NUMBER VALIDATION
# ============================================================================

def validate_phone_number(phone: str) -> bool:
    """
    Validate phone number format.
    
    Args:
        phone: Phone number to validate
        
    Returns:
        True if valid E.164 format
        
    Example:
        >>> validate_phone_number("+12125551234")
        True
        >>> validate_phone_number("212-555-1234")
        False
    """
    if not phone or not isinstance(phone, str):
        return False
    
    # Must start with +
    if not phone.startswith("+"):
        return False
    
    # Must have 10-15 digits after +
    digits = phone[1:].replace(" ", "").replace("-", "")
    if not digits.isdigit():
        return False
    
    if len(digits) < 10 or len(digits) > 15:
        return False
    
    return True


def format_phone_number(phone: str) -> str:
    """
    Format phone number to E.164 format.
    
    Args:
        phone: Phone number (various formats accepted)
        
    Returns:
        Phone number in E.164 format
        
    Raises:
        ValueError: If phone number is invalid
        
    Example:
        >>> format_phone_number("212-555-1234")
        "+12125551234"
    """
    if not phone:
        raise ValueError("Phone number cannot be empty")
    
    # Remove common formatting
    clean = phone.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    
    # Add + if missing
    if not clean.startswith("+"):
        # Assume US number if no country code
        if len(clean) == 10:
            clean = "+1" + clean
        else:
            clean = "+" + clean
    
    # Validate
    if not validate_phone_number(clean):
        raise ValueError(f"Invalid phone number format: {phone}")
    
    return clean


# ============================================================================
# MESSAGE FORMATTING
# ============================================================================

def format_order_confirmation_message(order_summary: Dict[str, Any]) -> str:
    """
    Format order confirmation message.
    
    Args:
        order_summary: Order details dictionary
        
    Returns:
        Formatted SMS message text
        
    Example:
        >>> order = {
        >>>     "order_id": "ORD123",
        >>>     "items": [{"name": "Pizza", "quantity": 1}],
        >>>     "total": 15.99
        >>> }
        >>> msg = format_order_confirmation_message(order)
    """
    # Extract order details
    order_id = order_summary.get("order_id", "N/A")
    restaurant_name = order_summary.get("restaurant_name", "Restaurant")
    items = order_summary.get("items", [])
    total = order_summary.get("total", 0.0)
    
    # Build message
    message = f"✅ Order Confirmed!\n\n"
    message += f"Order #{order_id}\n"
    message += f"{restaurant_name}\n\n"
    
    # Add items
    message += "Items:\n"
    for item in items:
        quantity = item.get("quantity", 1)
        name = item.get("name", "Unknown")
        message += f"• {quantity}x {name}\n"
    
    message += f"\nTotal: ${total:.2f}\n\n"
    
    # Add delivery/pickup info
    delivery_type = order_summary.get("type", "pickup")
    if delivery_type == "delivery":
        address = order_summary.get("delivery_address", "")
        if address:
            message += f"Delivery to: {address}\n"
        estimated_time = order_summary.get("estimated_time", "30-45 min")
        message += f"Estimated delivery: {estimated_time}\n"
    else:
        estimated_time = order_summary.get("estimated_time", "15-20 min")
        message += f"Ready for pickup in: {estimated_time}\n"
    
    message += f"\nThank you for your order!"
    
    return message


# ============================================================================
# SMS SENDING FUNCTIONS
# ============================================================================

def send_sms(
    to_phone: str,
    message: str,
    from_phone: Optional[str] = None,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Send SMS message via Twilio.
    
    Args:
        to_phone: Recipient phone number (E.164 format)
        message: Message text (max 1600 chars for concatenation)
        from_phone: Sender phone (defaults to TWILIO_PHONE_NUMBER)
        max_retries: Number of retry attempts on failure
        
    Returns:
        Dictionary with send result:
        {
            "success": bool,
            "message_sid": str,
            "status": str,
            "error": str (if failed)
        }
        
    Raises:
        ValueError: If phone number or message is invalid
        
    Example:
        >>> result = send_sms("+12125551234", "Hello!")
        >>> if result["success"]:
        >>>     print(f"Sent: {result['message_sid']}")
    """
    # Validate inputs
    if not message or not message.strip():
        raise ValueError("Message cannot be empty")
    
    if len(message) > 1600:
        raise ValueError(
            f"Message too long ({len(message)} chars). Max 1600 for SMS."
        )
    
    # Format phone number
    try:
        to_phone = format_phone_number(to_phone)
    except ValueError as e:
        raise ValueError(f"Invalid recipient phone: {str(e)}")
    
    # Get sender phone
    if not from_phone:
        from_phone = _get_twilio_phone()
    
    # Retry loop
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Send via Twilio
            client = _get_client()
            
            twilio_message = client.messages.create(
                body=message,
                from_=from_phone,
                to=to_phone
            )
            
            # Success
            logger.info(
                f"SMS sent successfully: {twilio_message.sid} "
                f"to {to_phone} (attempt {attempt + 1})"
            )
            
            return {
                "success": True,
                "message_sid": twilio_message.sid,
                "status": twilio_message.status,
                "to": to_phone,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except TwilioRestException as e:
            last_error = str(e)
            logger.warning(
                f"Twilio error (attempt {attempt + 1}/{max_retries}): {e.msg}"
            )
            
            # Don't retry on certain errors
            if e.code in [21211, 21614]:  # Invalid phone or blocked
                logger.error(f"Non-retryable error: {e.msg}")
                break
            
            # Wait before retry (except on last attempt)
            if attempt < max_retries - 1:
                import time
                time.sleep(1 * (attempt + 1))  # Exponential backoff
        
        except Exception as e:
            last_error = str(e)
            logger.error(f"Unexpected error sending SMS (attempt {attempt + 1}): {str(e)}")
            break
    
    # All retries failed
    logger.error(f"Failed to send SMS to {to_phone} after {max_retries} attempts")
    
    return {
        "success": False,
        "error": last_error,
        "to": to_phone,
        "timestamp": datetime.utcnow().isoformat()
    }


def send_order_confirmation(
    phone: str,
    order_summary: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Send order confirmation SMS to customer.
    
    Args:
        phone: Customer phone number
        order_summary: Order details dictionary containing:
            - order_id: str
            - restaurant_name: str (optional)
            - items: list of {name, quantity, price}
            - total: float
            - type: "delivery" or "pickup"
            - delivery_address: str (if delivery)
            - estimated_time: str (optional)
        
    Returns:
        Send result dictionary
        
    Example:
        >>> order = {
        >>>     "order_id": "ORD123",
        >>>     "items": [
        >>>         {"name": "Large Pizza", "quantity": 1, "price": 15.99}
        >>>     ],
        >>>     "total": 15.99,
        >>>     "type": "pickup",
        >>>     "estimated_time": "20 minutes"
        >>> }
        >>> result = send_order_confirmation("+12125551234", order)
    """
    try:
        # Format message
        message = format_order_confirmation_message(order_summary)
        
        # Send SMS
        result = send_sms(phone, message)
        
        if result["success"]:
            logger.info(
                f"Order confirmation sent for order "
                f"{order_summary.get('order_id', 'unknown')}"
            )
        else:
            logger.error(
                f"Failed to send order confirmation: {result.get('error')}"
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error sending order confirmation: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "to": phone,
            "timestamp": datetime.utcnow().isoformat()
        }


# ============================================================================
# ADDITIONAL NOTIFICATION FUNCTIONS
# ============================================================================

def send_order_ready(phone: str, order_id: str) -> Dict[str, Any]:
    """
    Notify customer that order is ready.
    
    Args:
        phone: Customer phone number
        order_id: Order identifier
        
    Returns:
        Send result dictionary
    """
    message = (
        f"🍕 Your order #{order_id} is ready for pickup!\n\n"
        f"Please come to the restaurant to collect your order.\n\n"
        f"Thank you!"
    )
    
    result = send_sms(phone, message)
    
    if result["success"]:
        logger.info(f"Order ready notification sent for {order_id}")
    
    return result


def send_delivery_update(
    phone: str,
    order_id: str,
    status: str,
    eta: Optional[str] = None
) -> Dict[str, Any]:
    """
    Send delivery status update.
    
    Args:
        phone: Customer phone number
        order_id: Order identifier
        status: Status message (e.g., "Out for delivery")
        eta: Estimated arrival time (optional)
        
    Returns:
        Send result dictionary
    """
    message = f"🚗 Order #{order_id} Update\n\n{status}"
    
    if eta:
        message += f"\n\nEstimated arrival: {eta}"
    
    message += "\n\nThank you for your patience!"
    
    result = send_sms(phone, message)
    
    if result["success"]:
        logger.info(f"Delivery update sent for {order_id}: {status}")
    
    return result


def send_custom_message(
    phone: str,
    message: str,
    restaurant_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Send custom message to customer.
    
    Args:
        phone: Customer phone number
        message: Custom message text
        restaurant_name: Restaurant name to prepend (optional)
        
    Returns:
        Send result dictionary
    """
    if restaurant_name:
        full_message = f"{restaurant_name}\n\n{message}"
    else:
        full_message = message
    
    return send_sms(phone, full_message)


# ============================================================================
# BULK MESSAGING
# ============================================================================

def send_bulk_sms(
    phone_numbers: List[str],
    message: str
) -> Dict[str, Any]:
    """
    Send same message to multiple recipients.
    
    Args:
        phone_numbers: List of phone numbers
        message: Message to send
        
    Returns:
        Bulk send results:
        {
            "total": int,
            "successful": int,
            "failed": int,
            "results": list of individual results
        }
    """
    results = []
    successful = 0
    failed = 0
    
    for phone in phone_numbers:
        result = send_sms(phone, message, max_retries=1)
        results.append(result)
        
        if result["success"]:
            successful += 1
        else:
            failed += 1
    
    logger.info(
        f"Bulk SMS completed: {successful} sent, {failed} failed "
        f"out of {len(phone_numbers)} total"
    )
    
    return {
        "total": len(phone_numbers),
        "successful": successful,
        "failed": failed,
        "results": results,
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_message_status(message_sid: str) -> Dict[str, Any]:
    """
    Check status of sent message.
    
    Args:
        message_sid: Twilio message SID
        
    Returns:
        Message status details
    """
    try:
        client = _get_client()
        message = client.messages(message_sid).fetch()
        
        return {
            "sid": message.sid,
            "status": message.status,
            "to": message.to,
            "from": message.from_,
            "date_sent": message.date_sent,
            "error_code": message.error_code,
            "error_message": message.error_message
        }
        
    except TwilioRestException as e:
        logger.error(f"Failed to fetch message status: {e.msg}")
        return {
            "error": str(e)
        }


def estimate_sms_segments(message: str) -> int:
    """
    Estimate number of SMS segments for message.
    
    Args:
        message: Message text
        
    Returns:
        Number of segments (each segment = 160 chars for ASCII, 70 for Unicode)
    """
    # Check if contains Unicode
    is_unicode = not all(ord(char) < 128 for char in message)
    
    if is_unicode:
        # Unicode SMS: 70 chars per segment
        segment_size = 70
        concat_size = 67  # For concatenated messages
    else:
        # GSM-7 encoding: 160 chars per segment
        segment_size = 160
        concat_size = 153  # For concatenated messages
    
    length = len(message)
    
    if length <= segment_size:
        return 1
    else:
        # Concatenated messages use slightly less per segment
        return (length + concat_size - 1) // concat_size


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the SMS module.
    """
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("SMS Module Example")
    print("=" * 50)
    
    # Example order confirmation
    print("\nExample 1: Order Confirmation")
    print("-" * 50)
    
    order = {
        "order_id": "ORD12345",
        "restaurant_name": "Pizza Palace",
        "items": [
            {"name": "Large Pepperoni Pizza", "quantity": 1, "price": 15.99},
            {"name": "Garlic Bread", "quantity": 2, "price": 4.99}
        ],
        "total": 25.97,
        "type": "delivery",
        "delivery_address": "123 Main St, Apt 4B",
        "estimated_time": "35-45 minutes"
    }
    
    # Format message (without sending)
    message = format_order_confirmation_message(order)
    print("Message preview:")
    print(message)
    print(f"\nEstimated segments: {estimate_sms_segments(message)}")
    
    # Uncomment to actually send:
    # result = send_order_confirmation("+12125551234", order)
    # print(f"\nSend result: {result}")
    
    # Example order ready notification
    print("\n" + "=" * 50)
    print("Example 2: Order Ready Notification")
    print("-" * 50)
    
    # Uncomment to send:
    # result = send_order_ready("+12125551234", "ORD12345")
    # print(f"Send result: {result}")
    
    # Phone number validation examples
    print("\n" + "=" * 50)
    print("Example 3: Phone Number Validation")
    print("-" * 50)
    
    test_numbers = [
        "+12125551234",      # Valid
        "212-555-1234",      # Valid (will be formatted)
        "(212) 555-1234",    # Valid (will be formatted)
        "12125551234",       # Valid (will be formatted)
        "invalid"            # Invalid
    ]
    
    for num in test_numbers:
        try:
            formatted = format_phone_number(num)
            valid = validate_phone_number(formatted)
            print(f"✓ {num:20s} → {formatted} (valid: {valid})")
        except ValueError as e:
            print(f"✗ {num:20s} → Error: {e}")
    
    print("\n" + "=" * 50)
    print("Ready to send SMS notifications!")
    print("Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER")
