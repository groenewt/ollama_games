"""Configuration constants for game theory package."""

import asyncio
import aiohttp
from typing import List, Tuple, Dict, Any

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

# Default settings
DEFAULT_TIMEOUT = 30
DEFAULT_NUM_GAMES = 10
MAX_NUM_GAMES = 100


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
    except Exception:
        return False


async def discover_models(endpoint: str, timeout: float = DEFAULT_TIMEOUT) -> List[str]:
    """Query an Ollama endpoint for available models.

    Args:
        endpoint: The Ollama API base URL.
        timeout: Request timeout in seconds.

    Returns:
        List of model names available at this endpoint, empty if failed.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{endpoint}/api/tags",
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("models", [])
                    return [m.get("name", "") for m in models if m.get("name")]
                return []
    except Exception:
        return []


async def discover_all_available(
    endpoints: List[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """Discover all available models across all endpoints.

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

    # Query all endpoints in parallel
    tasks = [discover_models(ep, timeout) for ep in endpoints]
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


def discover_sync(
    endpoints: List[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """Synchronous wrapper for discover_all_available.

    Args:
        endpoints: List of endpoints to check.
        timeout: Timeout per endpoint in seconds.

    Returns:
        Same as discover_all_available.
    """
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(discover_all_available(endpoints, timeout))
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(discover_all_available(endpoints, timeout))
    except Exception:
        # Fall back to defaults on any error
        return DEFAULT_OLLAMA_MODELS.copy(), [], {}
