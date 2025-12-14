"""Factory for creating Hamilton drivers with Burr integration."""
from typing import Optional
import os

from hamilton import driver


def create_game_driver(
    game_type: str,
    tracer=None,
    enable_hamilton_ui: bool = False,
) -> driver.Driver:
    """Create Hamilton driver configured for game type.

    Args:
        game_type: Type of game (discrete, allocation, permutation)
        tracer: Optional Burr tracer for span integration
        enable_hamilton_ui: Whether to enable Hamilton tracking

    Returns:
        Configured Hamilton Driver
    """
    from . import prompt_building, llm_execution, response_parsing
    from .tracer import HamiltonBurrTracer

    builder = driver.Builder()
    builder = builder.with_config({"game_type": game_type})
    builder = builder.with_modules(
        prompt_building,
        llm_execution,
        response_parsing,
    )

    adapters = []

    if tracer:
        hamilton_tracer = HamiltonBurrTracer(tracer)
        adapters.append(hamilton_tracer)

    if adapters:
        builder = builder.with_adapters(*adapters)

    return builder.build()
