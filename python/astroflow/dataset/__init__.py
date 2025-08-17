"""
AstroFlow Dataset Module

This module provides utilities for generating and handling astronomical datasets,
particularly for FRB (Fast Radio Burst) detection and classification.
"""

from .simulator import SimulationConfig, generate_synthetic_dataset

__all__ = [
    'SimulationConfig',
    'generate_synthetic_dataset',
]
