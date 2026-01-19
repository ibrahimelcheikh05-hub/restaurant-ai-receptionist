"""
Configuration Module
====================
Centralized environment variable loading, validation, and access.
Validates all required configuration at startup to fail fast.

NO BUSINESS LOGIC - Pure configuration management only.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv


# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# ENVIRONMENT LOADING
# ============================================================================

def load_environment():
    """
    Load environment variables from .env file if present.
    Safe to call multiple times.
    """
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)
        logger.info("Loaded environment from .env file")
    else:
        logger.info("No .env file found, using system environment variables")


# Load on module import
load_environment()


# ============================================================================
# CONFIGURATION EXCEPTION
# ============================================================================

class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def _get_required_env(key: str, description: str = None) -> str:
    """
    Get required environment variable.
    
    Args:
        key: Environment variable name
        description: Optional description for error message
        
    Returns:
        Environment variable value
        
    Raises:
        ConfigurationError: If variable is missing or empty
    """
    value = os.getenv(key)
    
    if not value or value.strip() == "":
        desc = f" ({description})" if description else ""
        raise ConfigurationError(
            f"Missing required environment variable: {key}{desc}"
        )
    
    return value.strip()


def _get_optional_env(key: str, default: str = None) -> Optional[str]:
    """
    Get optional environment variable.
    
    Args:
        key: Environment variable name
        default: Default value if not set
        
    Returns:
        Environment variable value or default
    """
    value = os.getenv(key, default)
    return value.strip() if value else default


def _get_bool_env(key: str, default: bool = False) -> bool:
    """
    Get boolean environment variable.
    
    Args:
        key: Environment variable name
        default: Default value if not set
        
    Returns:
        Boolean value
    """
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on", "enabled")


def _get_int_env(key: str, default: int = None) -> Optional[int]:
    """
    Get integer environment variable.
    
    Args:
        key: Environment variable name
        default: Default value if not set
        
    Returns:
        Integer value or None
        
    Raises:
        ConfigurationError: If value is not a valid integer
    """
    value = os.getenv(key)
    
    if not value:
        return default
    
    try:
        return int(value)
    except ValueError:
        raise ConfigurationError(
            f"Invalid integer value for {key}: {value}"
        )


# ============================================================================
# CORE CONFIGURATION
# ============================================================================

class CoreConfig:
    """Core system configuration."""
    
    def __init__(self):
        # System prompt for AI agent
        self.system_agent_prompt = _get_required_env(
            "SYSTEM_AGENT_PROMPT",
            "AI agent system prompt"
        )
        
        # LLM Provider
        self.llm_provider = _get_required_env(
            "LLM_PROVIDER",
            "AI provider: 'claude' or 'openai'"
        ).lower()
        
        if self.llm_provider not in ["claude", "openai"]:
            raise ConfigurationError(
                f"Invalid LLM_PROVIDER: {self.llm_provider}. "
                f"Must be 'claude' or 'openai'"
            )
        
        # LLM API Key (provider-specific)
        if self.llm_provider == "claude":
            self.llm_api_key = _get_required_env(
                "ANTHROPIC_API_KEY",
                "Anthropic Claude API key"
            )
            self.llm_model = _get_optional_env(
                "CLAUDE_MODEL",
                "claude-sonnet-4-20250514"
            )
        else:  # openai
            self.llm_api_key = _get_required_env(
                "OPENAI_API_KEY",
                "OpenAI API key"
            )
            self.llm_model = _get_optional_env(
                "OPENAI_MODEL",
                "gpt-4"
            )
        
        # Restaurant configuration
        self.default_restaurant_id = _get_optional_env(
            "DEFAULT_RESTAURANT_ID",
            "rest_001"
        )


# ============================================================================
# GOOGLE CLOUD CONFIGURATION
# ============================================================================

class GoogleCloudConfig:
    """Google Cloud services configuration."""
    
    def __init__(self):
        # Credentials (path to JSON file OR JSON string)
        self.credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.credentials_json = os.getenv("GOOGLE_CLOUD_CREDENTIALS_JSON")
        
        if not self.credentials_path and not self.credentials_json:
            raise ConfigurationError(
                "Missing Google Cloud credentials. Set either "
                "GOOGLE_APPLICATION_CREDENTIALS (path) or "
                "GOOGLE_CLOUD_CREDENTIALS_JSON (JSON string)"
            )
        
        # Validate credentials file exists if path provided
        if self.credentials_path:
            if not Path(self.credentials_path).exists():
                raise ConfigurationError(
                    f"Google credentials file not found: {self.credentials_path}"
                )
        
        # Project ID (optional, can be inferred from credentials)
        self.project_id = _get_optional_env("GOOGLE_PROJECT_ID")
        
        # API-specific settings
        self.stt_language_codes = ["en-US", "ar-SA", "es-ES"]
        self.tts_sample_rate = _get_int_env("TTS_SAMPLE_RATE", 16000)
        self.stt_sample_rate = _get_int_env("STT_SAMPLE_RATE", 16000)


# ============================================================================
# TWILIO CONFIGURATION
# ============================================================================

class TwilioConfig:
    """Twilio telephony configuration."""
    
    def __init__(self):
        self.account_sid = _get_required_env(
            "TWILIO_ACCOUNT_SID",
            "Twilio Account SID"
        )
        
        self.auth_token = _get_required_env(
            "TWILIO_AUTH_TOKEN",
            "Twilio Auth Token"
        )
        
        self.phone_number = _get_required_env(
            "TWILIO_PHONE_NUMBER",
            "Twilio phone number (E.164 format)"
        )
        
        # Validate phone number format
        if not self.phone_number.startswith("+"):
            raise ConfigurationError(
                f"TWILIO_PHONE_NUMBER must be in E.164 format (start with +): "
                f"{self.phone_number}"
            )
        
        # Call transfer configuration
        self.human_transfer_number = _get_optional_env(
            "HUMAN_TRANSFER_NUMBER"
        )
        
        # Validate transfer number format if provided
        if self.human_transfer_number:
            if not self.human_transfer_number.startswith("+"):
                raise ConfigurationError(
                    f"HUMAN_TRANSFER_NUMBER must be in E.164 format (start with +): "
                    f"{self.human_transfer_number}"
                )
        
        # Optional: Recording storage URL
        self.recording_status_callback = _get_optional_env(
            "TWILIO_RECORDING_STATUS_CALLBACK"
        )


# ============================================================================
# SUPABASE CONFIGURATION
# ============================================================================

class SupabaseConfig:
    """Supabase database configuration."""
    
    def __init__(self):
        self.url = _get_required_env(
            "SUPABASE_URL",
            "Supabase project URL"
        )
        
        self.key = _get_required_env(
            "SUPABASE_KEY",
            "Supabase anon or service role key"
        )
        
        # Validate URL format
        if not self.url.startswith("https://"):
            raise ConfigurationError(
                f"SUPABASE_URL must start with https://: {self.url}"
            )
        
        # Connection settings
        self.max_connections = _get_int_env("SUPABASE_MAX_CONNECTIONS", 10)
        self.connection_timeout = _get_int_env("SUPABASE_TIMEOUT", 30)


# ============================================================================
# VOICE CONFIGURATION
# ============================================================================

class VoiceConfig:
    """Voice synthesis and recognition configuration."""
    
    def __init__(self):
        # Default language
        self.default_language = _get_optional_env(
            "DEFAULT_LANGUAGE",
            "en"
        )
        
        if self.default_language not in ["en", "ar", "es"]:
            raise ConfigurationError(
                f"Invalid DEFAULT_LANGUAGE: {self.default_language}. "
                f"Must be 'en', 'ar', or 'es'"
            )
        
        # Voice names for each language (Google Cloud TTS)
        self.voice_en = _get_optional_env(
            "VOICE_EN",
            "en-US-Neural2-F"
        )
        
        self.voice_ar = _get_optional_env(
            "VOICE_AR",
            "ar-XA-Standard-A"
        )
        
        self.voice_es = _get_optional_env(
            "VOICE_ES",
            "es-ES-Neural2-A"
        )
        
        # Voice library mapping
        self.voice_library = {
            "en": self.voice_en,
            "ar": self.voice_ar,
            "es": self.voice_es
        }
        
        # Audio settings
        self.speaking_rate = float(_get_optional_env("SPEAKING_RATE", "1.0"))
        self.pitch = float(_get_optional_env("PITCH", "0.0"))
        self.volume_gain_db = float(_get_optional_env("VOLUME_GAIN_DB", "0.0"))
        
        # Validate ranges
        if not 0.25 <= self.speaking_rate <= 4.0:
            raise ConfigurationError(
                f"SPEAKING_RATE must be between 0.25 and 4.0: {self.speaking_rate}"
            )
        
        if not -20.0 <= self.pitch <= 20.0:
            raise ConfigurationError(
                f"PITCH must be between -20.0 and 20.0: {self.pitch}"
            )
        
        if not -96.0 <= self.volume_gain_db <= 16.0:
            raise ConfigurationError(
                f"VOLUME_GAIN_DB must be between -96.0 and 16.0: {self.volume_gain_db}"
            )


# ============================================================================
# FEATURE FLAGS
# ============================================================================

class FeatureFlags:
    """Feature flags for optional functionality."""
    
    def __init__(self):
        self.enable_barge_in = _get_bool_env("ENABLE_BARGE_IN", True)
        self.enable_call_recording = _get_bool_env("ENABLE_CALL_RECORDING", False)
        self.enable_call_transfer = _get_bool_env("ENABLE_CALL_TRANSFER", False)
        self.enable_translation_cache = _get_bool_env("ENABLE_TRANSLATION_CACHE", True)
        self.enable_upsells = _get_bool_env("ENABLE_UPSELLS", True)
        
        # Audio processing
        self.enable_noise_reduction = _get_bool_env("ENABLE_NOISE_REDUCTION", True)
        self.enable_automatic_punctuation = _get_bool_env(
            "ENABLE_AUTOMATIC_PUNCTUATION",
            True
        )
        
        # Development/debug features
        self.debug_mode = _get_bool_env("DEBUG_MODE", False)
        self.log_transcripts = _get_bool_env("LOG_TRANSCRIPTS", True)


# ============================================================================
# SERVER CONFIGURATION
# ============================================================================

class ServerConfig:
    """Web server configuration."""
    
    def __init__(self):
        self.host = _get_optional_env("HOST", "0.0.0.0")
        self.port = _get_int_env("PORT", 8000)
        self.base_url = _get_optional_env(
            "BASE_URL",
            f"http://{self.host}:{self.port}"
        )
        
        # Validate base URL
        if not self.base_url.startswith(("http://", "https://")):
            raise ConfigurationError(
                f"BASE_URL must start with http:// or https://: {self.base_url}"
            )
        
        # CORS settings
        self.cors_origins = _get_optional_env("CORS_ORIGINS", "*").split(",")
        
        # Timeouts
        self.request_timeout = _get_int_env("REQUEST_TIMEOUT", 60)
        self.websocket_timeout = _get_int_env("WEBSOCKET_TIMEOUT", 300)
        
        # Logging
        self.log_level = _get_optional_env("LOG_LEVEL", "INFO").upper()
        
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ConfigurationError(
                f"Invalid LOG_LEVEL: {self.log_level}"
            )


# ============================================================================
# MAIN CONFIGURATION CLASS
# ============================================================================

class Config:
    """
    Main configuration container.
    Loads and validates all configuration on initialization.
    """
    
    def __init__(self):
        """
        Initialize and validate all configuration.
        
        Raises:
            ConfigurationError: If any required configuration is missing or invalid
        """
        try:
            self.core = CoreConfig()
            self.google = GoogleCloudConfig()
            self.twilio = TwilioConfig()
            self.supabase = SupabaseConfig()
            self.voice = VoiceConfig()
            self.features = FeatureFlags()
            self.server = ServerConfig()
            
            logger.info("Configuration loaded and validated successfully")
            
        except ConfigurationError as e:
            logger.error(f"Configuration error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {str(e)}")
            raise ConfigurationError(f"Configuration initialization failed: {str(e)}")
    
    def get_safe_summary(self) -> Dict[str, Any]:
        """
        Get safe configuration summary (no secrets).
        
        Returns:
            Dictionary with non-sensitive configuration
        """
        return {
            "llm_provider": self.core.llm_provider,
            "llm_model": self.core.llm_model,
            "default_restaurant": self.core.default_restaurant_id,
            "default_language": self.voice.default_language,
            "supported_languages": list(self.voice.voice_library.keys()),
            "features": {
                "barge_in": self.features.enable_barge_in,
                "recording": self.features.enable_call_recording,
                "transfer": self.features.enable_call_transfer,
                "upsells": self.features.enable_upsells,
            },
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "log_level": self.server.log_level,
            },
            "audio": {
                "sample_rate": self.google.tts_sample_rate,
                "speaking_rate": self.voice.speaking_rate,
                "noise_reduction": self.features.enable_noise_reduction,
            }
        }
    
    def validate_runtime_dependencies(self) -> List[str]:
        """
        Validate that runtime dependencies are accessible.
        
        Returns:
            List of warnings (empty if all OK)
        """
        warnings = []
        
        # Check Google credentials file
        if self.google.credentials_path:
            if not Path(self.google.credentials_path).exists():
                warnings.append(
                    f"Google credentials file not found: {self.google.credentials_path}"
                )
        
        return warnings


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get global configuration instance.
    Initializes on first call.
    
    Returns:
        Config instance
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    global _config
    
    if _config is None:
        _config = Config()
    
    return _config


def reload_config():
    """
    Reload configuration from environment.
    Useful for testing or dynamic reconfiguration.
    """
    global _config
    load_environment()
    _config = Config()
    logger.info("Configuration reloaded")


# ============================================================================
# CONVENIENCE GETTERS
# ============================================================================

def get_llm_provider() -> str:
    """Get LLM provider name."""
    return get_config().core.llm_provider


def get_llm_api_key() -> str:
    """Get LLM API key."""
    return get_config().core.llm_api_key


def get_llm_model() -> str:
    """Get LLM model name."""
    return get_config().core.llm_model


def get_system_prompt() -> str:
    """Get system agent prompt."""
    return get_config().core.system_agent_prompt


def is_feature_enabled(feature_name: str) -> bool:
    """
    Check if a feature is enabled.
    
    Args:
        feature_name: Feature flag name
        
    Returns:
        True if enabled
    """
    features = get_config().features
    return getattr(features, f"enable_{feature_name}", False)


def get_voice_for_language(language: str) -> str:
    """
    Get voice name for language.
    
    Args:
        language: Language code ('en', 'ar', 'es')
        
    Returns:
        Voice name
    """
    return get_config().voice.voice_library.get(
        language,
        get_config().voice.voice_en
    )


# ============================================================================
# VALIDATION FUNCTION
# ============================================================================

def validate_configuration():
    """
    Validate configuration and print summary.
    Useful for startup checks.
    
    Raises:
        ConfigurationError: If configuration is invalid
    """
    config = get_config()
    
    # Get safe summary
    summary = config.get_safe_summary()
    
    logger.info("Configuration Summary:")
    logger.info(f"  LLM Provider: {summary['llm_provider']}")
    logger.info(f"  LLM Model: {summary['llm_model']}")
    logger.info(f"  Default Language: {summary['default_language']}")
    logger.info(f"  Supported Languages: {', '.join(summary['supported_languages'])}")
    logger.info(f"  Server: {summary['server']['host']}:{summary['server']['port']}")
    logger.info(f"  Log Level: {summary['server']['log_level']}")
    
    logger.info("Feature Flags:")
    for feature, enabled in summary['features'].items():
        status = "enabled" if enabled else "disabled"
        logger.info(f"  {feature}: {status}")
    
    # Check runtime dependencies
    warnings = config.validate_runtime_dependencies()
    if warnings:
        logger.warning("Configuration warnings:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
    
    logger.info("Configuration validation complete")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage and configuration testing.
    """
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Configuration Module Test")
    print("=" * 50)
    
    try:
        # Validate configuration
        validate_configuration()
        
        # Test convenience getters
        print("\nConvenience Getters:")
        print(f"LLM Provider: {get_llm_provider()}")
        print(f"LLM Model: {get_llm_model()}")
        print(f"Barge-in enabled: {is_feature_enabled('barge_in')}")
        print(f"Voice for English: {get_voice_for_language('en')}")
        print(f"Voice for Arabic: {get_voice_for_language('ar')}")
        
        # Get full config
        config = get_config()
        
        print("\nDetailed Configuration:")
        print(f"Twilio Phone: {config.twilio.phone_number}")
        print(f"Supabase URL: {config.supabase.url[:30]}...")
        print(f"Default Restaurant: {config.core.default_restaurant_id}")
        
        print("\n✓ Configuration is valid!")
        
    except ConfigurationError as e:
        print(f"\n✗ Configuration Error: {e}")
        exit(1)
    
    except Exception as e:
        print(f"\n✗ Unexpected Error: {e}")
        exit(1)
