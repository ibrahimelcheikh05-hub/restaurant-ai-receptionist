"""
Supabase Data Layer
===================
All database interactions for the restaurant voice ordering system.
Pure data gateway - no business logic.
"""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SupabaseDB:
    """
    Supabase database gateway for restaurant voice ordering system.
    Handles all data persistence operations.
    """
    
    def __init__(self):
        """Initialize Supabase client with credentials from environment."""
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError(
                "Missing Supabase credentials. "
                "Set SUPABASE_URL and SUPABASE_KEY environment variables."
            )
        
        self.client: Client = create_client(supabase_url, supabase_key)
    
    # ========================================================================
    # MENU OPERATIONS
    # ========================================================================
    
    def fetch_menu(self, restaurant_id: str) -> List[Dict[str, Any]]:
        """
        Fetch menu items for a restaurant.
        
        Args:
            restaurant_id: Unique identifier for the restaurant
            
        Returns:
            List of menu item dictionaries
            
        Raises:
            Exception: If database query fails
        """
        try:
            response = self.client.table("menu_items").select("*").eq(
                "restaurant_id", restaurant_id
            ).eq("active", True).execute()
            
            return response.data
        except Exception as e:
            raise Exception(f"Failed to fetch menu: {str(e)}")
    
    # ========================================================================
    # ORDER OPERATIONS
    # ========================================================================
    
    def store_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store a new order in the database.
        
        Args:
            order_data: Dictionary containing order information
                Required fields:
                - restaurant_id: str
                - customer_phone: str
                - items: List[Dict] (each with item_id, quantity, customizations)
                - total_amount: float
                - status: str (e.g., 'pending', 'confirmed')
                Optional fields:
                - customer_name: str
                - delivery_address: str
                - special_instructions: str
                - call_log_id: str (reference to call that created this order)
                
        Returns:
            Created order record with generated ID
            
        Raises:
            Exception: If database insert fails
        """
        try:
            # Add timestamp
            order_data["created_at"] = datetime.utcnow().isoformat()
            
            response = self.client.table("orders").insert(order_data).execute()
            
            return response.data[0] if response.data else {}
        except Exception as e:
            raise Exception(f"Failed to store order: {str(e)}")
    
    def update_order_status(
        self, 
        order_id: str, 
        status: str, 
        updated_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update the status of an existing order.
        
        Args:
            order_id: Unique identifier for the order
            status: New status value
            updated_by: Optional identifier of who updated the order
            
        Returns:
            Updated order record
            
        Raises:
            Exception: If database update fails
        """
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if updated_by:
                update_data["updated_by"] = updated_by
            
            response = self.client.table("orders").update(
                update_data
            ).eq("id", order_id).execute()
            
            return response.data[0] if response.data else {}
        except Exception as e:
            raise Exception(f"Failed to update order status: {str(e)}")
    
    # ========================================================================
    # CALL LOG OPERATIONS
    # ========================================================================
    
    def store_call_log(self, call_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store a call log record.
        
        Args:
            call_data: Dictionary containing call information
                Required fields:
                - restaurant_id: str
                - caller_phone: str
                - call_sid: str (Twilio call identifier)
                - direction: str ('inbound' or 'outbound')
                Optional fields:
                - duration: int (seconds)
                - status: str (e.g., 'completed', 'no-answer', 'busy')
                - transcript: str
                - recording_url: str
                - agent_id: str
                
        Returns:
            Created call log record with generated ID
            
        Raises:
            Exception: If database insert fails
        """
        try:
            # Add timestamp
            call_data["created_at"] = datetime.utcnow().isoformat()
            
            response = self.client.table("call_logs").insert(call_data).execute()
            
            return response.data[0] if response.data else {}
        except Exception as e:
            raise Exception(f"Failed to store call log: {str(e)}")
    
    def update_call_log(
        self, 
        call_log_id: str, 
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update an existing call log record.
        
        Args:
            call_log_id: Unique identifier for the call log
            updates: Dictionary of fields to update
            
        Returns:
            Updated call log record
            
        Raises:
            Exception: If database update fails
        """
        try:
            updates["updated_at"] = datetime.utcnow().isoformat()
            
            response = self.client.table("call_logs").update(
                updates
            ).eq("id", call_log_id).execute()
            
            return response.data[0] if response.data else {}
        except Exception as e:
            raise Exception(f"Failed to update call log: {str(e)}")
    
    # ========================================================================
    # RECORDING OPERATIONS
    # ========================================================================
    
    def store_recording_path(
        self, 
        call_log_id: str, 
        recording_url: str,
        duration: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Store recording URL/path for a call.
        
        Args:
            call_log_id: ID of the call log to update
            recording_url: URL or path to the recording
            duration: Optional recording duration in seconds
            
        Returns:
            Updated call log record
            
        Raises:
            Exception: If database update fails
        """
        try:
            update_data = {
                "recording_url": recording_url,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if duration is not None:
                update_data["duration"] = duration
            
            response = self.client.table("call_logs").update(
                update_data
            ).eq("id", call_log_id).execute()
            
            return response.data[0] if response.data else {}
        except Exception as e:
            raise Exception(f"Failed to store recording path: {str(e)}")
    
    # ========================================================================
    # RESTAURANT SETTINGS OPERATIONS
    # ========================================================================
    
    def fetch_restaurant_settings(
        self, 
        restaurant_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch settings/configuration for a restaurant.
        
        Args:
            restaurant_id: Unique identifier for the restaurant
            
        Returns:
            Restaurant settings dictionary or None if not found
            Settings may include:
            - name: str
            - phone: str
            - address: str
            - business_hours: Dict
            - delivery_enabled: bool
            - pickup_enabled: bool
            - tax_rate: float
            - ai_agent_config: Dict
            - payment_methods: List[str]
            
        Raises:
            Exception: If database query fails
        """
        try:
            response = self.client.table("restaurants").select("*").eq(
                "id", restaurant_id
            ).execute()
            
            return response.data[0] if response.data else None
        except Exception as e:
            raise Exception(f"Failed to fetch restaurant settings: {str(e)}")
    
    def update_restaurant_settings(
        self,
        restaurant_id: str,
        settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update restaurant settings.
        
        Args:
            restaurant_id: Unique identifier for the restaurant
            settings: Dictionary of settings to update
            
        Returns:
            Updated restaurant record
            
        Raises:
            Exception: If database update fails
        """
        try:
            settings["updated_at"] = datetime.utcnow().isoformat()
            
            response = self.client.table("restaurants").update(
                settings
            ).eq("id", restaurant_id).execute()
            
            return response.data[0] if response.data else {}
        except Exception as e:
            raise Exception(f"Failed to update restaurant settings: {str(e)}")
    
    # ========================================================================
    # CUSTOMER OPERATIONS (BONUS)
    # ========================================================================
    
    def fetch_customer_by_phone(
        self, 
        phone: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch customer record by phone number.
        
        Args:
            phone: Customer phone number
            
        Returns:
            Customer record or None if not found
            
        Raises:
            Exception: If database query fails
        """
        try:
            response = self.client.table("customers").select("*").eq(
                "phone", phone
            ).execute()
            
            return response.data[0] if response.data else None
        except Exception as e:
            raise Exception(f"Failed to fetch customer: {str(e)}")
    
    def store_customer(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store or update customer information.
        
        Args:
            customer_data: Dictionary containing customer information
                Fields: phone, name, email, address, preferences, etc.
                
        Returns:
            Customer record (created or updated)
            
        Raises:
            Exception: If database operation fails
        """
        try:
            customer_data["updated_at"] = datetime.utcnow().isoformat()
            
            # Upsert: insert or update if phone exists
            response = self.client.table("customers").upsert(
                customer_data,
                on_conflict="phone"
            ).execute()
            
            return response.data[0] if response.data else {}
        except Exception as e:
            raise Exception(f"Failed to store customer: {str(e)}")
    
    # ========================================================================
    # ANALYTICS & REPORTING (BONUS)
    # ========================================================================
    
    def fetch_orders_by_date_range(
        self,
        restaurant_id: str,
        start_date: str,
        end_date: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch orders within a date range for analytics.
        
        Args:
            restaurant_id: Unique identifier for the restaurant
            start_date: ISO format date string
            end_date: ISO format date string
            
        Returns:
            List of order records
            
        Raises:
            Exception: If database query fails
        """
        try:
            response = self.client.table("orders").select("*").eq(
                "restaurant_id", restaurant_id
            ).gte("created_at", start_date).lte(
                "created_at", end_date
            ).execute()
            
            return response.data
        except Exception as e:
            raise Exception(f"Failed to fetch orders by date range: {str(e)}")


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

# Create a singleton instance for easy importing
db = SupabaseDB()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the SupabaseDB class.
    """
    
    # Initialize
    database = SupabaseDB()
    
    # Fetch menu
    try:
        menu = database.fetch_menu(restaurant_id="rest_123")
        print(f"Menu items: {len(menu)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Store order
    order = {
        "restaurant_id": "rest_123",
        "customer_phone": "+1234567890",
        "customer_name": "John Doe",
        "items": [
            {
                "item_id": "item_1",
                "quantity": 2,
                "customizations": ["no onions"]
            }
        ],
        "total_amount": 25.50,
        "status": "pending"
    }
    
    try:
        created_order = database.store_order(order)
        print(f"Order created: {created_order.get('id')}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Store call log
    call_log = {
        "restaurant_id": "rest_123",
        "caller_phone": "+1234567890",
        "call_sid": "CA1234567890",
        "direction": "inbound",
        "status": "in-progress"
    }
    
    try:
        created_log = database.store_call_log(call_log)
        print(f"Call log created: {created_log.get('id')}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Fetch restaurant settings
    try:
        settings = database.fetch_restaurant_settings("rest_123")
        print(f"Restaurant: {settings.get('name') if settings else 'Not found'}")
    except Exception as e:
        print(f"Error: {e}")
