"""Configuration constants for game theory package."""

from typing import List

# Available Ollama models for AI players
OLLAMA_MODELS: List[str] = [
    "tinyllama:1.1b",
    "qwen3:0.6b",
    "granite3.1-moe:1b",
    "granite3.1-moe:3b",
    "gpt-oss:20b-cloud",
]

# Available Ollama API endpoints
OLLAMA_ENDPOINTS: List[str] = [
    "http://localhost:11434",
    "http://ollamaone:11434",
    "http://ollamatwo:11434",
    "http://host.containers.internal:11434",
]

# Default settings
DEFAULT_TIMEOUT = 30
DEFAULT_NUM_GAMES = 10
MAX_NUM_GAMES = 100
