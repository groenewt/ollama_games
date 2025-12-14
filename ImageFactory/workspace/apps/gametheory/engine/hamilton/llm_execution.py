"""Hamilton DAG for LLM execution."""
from typing import Any, Dict, Optional
import aiohttp


def request_payload(
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
) -> Dict[str, Any]:
    """Build Ollama API request payload."""
    return {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
        }
    }


def llm_endpoint_url(endpoint: str) -> str:
    """Construct full API URL."""
    return f"{endpoint}/api/generate"


async def raw_llm_response(
    http_session: aiohttp.ClientSession,
    llm_endpoint_url: str,
    request_payload: Dict[str, Any],
    timeout_seconds: float = 60.0,
) -> Dict[str, Any]:
    """Execute async HTTP call to Ollama."""
    try:
        async with http_session.post(
            llm_endpoint_url,
            json=request_payload,
            timeout=aiohttp.ClientTimeout(total=timeout_seconds),
        ) as response:
            if response.status == 200:
                return await response.json()
            return {"error": f"HTTP {response.status}", "response": ""}
    except Exception as e:
        return {"error": str(e), "response": ""}


def response_text(raw_llm_response: Dict[str, Any]) -> str:
    """Extract response text."""
    return raw_llm_response.get("response", "")


def prompt_tokens(raw_llm_response: Dict[str, Any]) -> Optional[int]:
    """Extract prompt token count."""
    return raw_llm_response.get("prompt_eval_count")


def completion_tokens(raw_llm_response: Dict[str, Any]) -> Optional[int]:
    """Extract completion token count."""
    return raw_llm_response.get("eval_count")
