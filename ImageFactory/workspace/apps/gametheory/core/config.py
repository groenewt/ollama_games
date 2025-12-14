"""Configuration constants for game theory package."""

import asyncio
import logging
import os
import aiohttp
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

# Default Ollama models (fallback if discovery fails)
DEFAULT_OLLAMA_MODELS: List[str] = [
    "tinyllama:1.1b",
    "qwen3:0.6b",
    "granite3.1-moe:1b",
    "granite3.1-moe:3b",
    "gpt-oss:20b-cloud",
]

# Default Ollama API endpoints to try
DEFAULT_OLLAMA_ENDPOINTS: List[str] = [
    "http://localhost:11434",
    "http://ollamaone:11434",
    "http://ollamatwo:11434",
    "http://host.docker.internal:11434",
]

# Legacy aliases for backwards compatibility
OLLAMA_MODELS = DEFAULT_OLLAMA_MODELS
OLLAMA_ENDPOINTS = DEFAULT_OLLAMA_ENDPOINTS

# Timeouts - configurable via environment variables
# LLM inference timeout (CPU models can be slow)
DEFAULT_TIMEOUT = int(os.environ.get("GAMETHEORY_TIMEOUT", "60"))
# Discovery timeout (fast check if endpoint is up)
DISCOVERY_TIMEOUT = int(os.environ.get("GAMETHEORY_DISCOVERY_TIMEOUT", "10"))
# Series-level timeout (max time for entire game series)
SERIES_TIMEOUT = int(os.environ.get("GAMETHEORY_SERIES_TIMEOUT", "600"))

# Game limits
DEFAULT_NUM_GAMES = 10
MAX_NUM_GAMES = int(os.environ.get("GAMETHEORY_MAX_GAMES", "1000"))

# Burr tracking configuration
BURR_TRACKING_ENABLED = os.environ.get("GAMETHEORY_BURR_TRACKING", "true").lower() == "true"
BURR_TRACKER_HOST = os.environ.get("BURR_TRACKER_HOST", "localhost")
BURR_TRACKER_PORT = int(os.environ.get("BURR_TRACKER_PORT", "7241"))


async def check_endpoint_available(endpoint: str, timeout: float = DEFAULT_TIMEOUT ) -> bool:
    """Check if an Ollama endpoint is reachable.

    Args:
        endpoint: The Ollama API base URL.
        timeout: Connection timeout in seconds.

    Returns:
        True if endpoint responds, False otherwise.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{endpoint}/api/tags",
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                return response.status == 200
    except asyncio.TimeoutError:
        logger.debug("Endpoint %s timed out after %.1fs", endpoint, timeout)
        return False
    except Exception as e:
        logger.debug("Endpoint %s check failed: %s: %s", endpoint, type(e).__name__, e)
        return False


async def discover_models(
    endpoint: str,
    timeout: float = DISCOVERY_TIMEOUT,
    session: aiohttp.ClientSession = None,
) -> List[str]:
    """Query an Ollama endpoint for available models.

    Args:
        endpoint: The Ollama API base URL.
        timeout: Request timeout in seconds.
        session: Optional shared aiohttp session for connection pooling.

    Returns:
        List of model names available at this endpoint, empty if failed.
    """
    should_close = session is None
    if session is None:
        session = aiohttp.ClientSession()

    try:
        async with session.get(
            f"{endpoint}/api/tags",
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            if response.status == 200:
                data = await response.json()
                models = data.get("models", [])
                model_names = [m.get("name", "") for m in models if m.get("name")]
                logger.debug("Discovered %d models at %s", len(model_names), endpoint)
                return model_names
            logger.debug("Endpoint %s returned status %d", endpoint, response.status)
            return []
    except asyncio.TimeoutError:
        logger.debug("Discovery timed out for %s after %.1fs", endpoint, timeout)
        return []
    except Exception as e:
        logger.debug("Discovery failed for %s: %s: %s", endpoint, type(e).__name__, e)
        return []
    finally:
        if should_close:
            await session.close()


async def discover_all_available(
    endpoints: List[str] = None,
    timeout: float = DISCOVERY_TIMEOUT,
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """Discover all available models across all endpoints.

    Uses connection pooling to efficiently query multiple endpoints in parallel.

    Args:
        endpoints: List of endpoints to check (defaults to DEFAULT_OLLAMA_ENDPOINTS).
        timeout: Timeout per endpoint in seconds.

    Returns:
        Tuple of:
        - available_models: Deduplicated list of all discovered models
        - available_endpoints: List of endpoints that responded
        - endpoint_models: Dict mapping endpoint -> list of its models
    """
    endpoints = endpoints or DEFAULT_OLLAMA_ENDPOINTS

    # Use shared session with connection pooling for all parallel requests
    connector = aiohttp.TCPConnector(limit=20, keepalive_timeout=30)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Query all endpoints in parallel with shared session
        tasks = [discover_models(ep, timeout, session) for ep in endpoints]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    available_endpoints = []
    all_models = set()
    endpoint_models = {}

    for endpoint, result in zip(endpoints, results):
        if isinstance(result, list) and result:
            available_endpoints.append(endpoint)
            endpoint_models[endpoint] = result
            all_models.update(result)

    # Sort models alphabetically
    available_models = sorted(all_models)

    return available_models, available_endpoints, endpoint_models
