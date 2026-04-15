"""
Ideia 2: Feature Windows + ILP with Global Sampling

A rapid baseline approach to validate ILP viability with limited dimensionality,
without depending on clustering.

Key components:
- Pipeline: orchestrates the full workflow
- DataLoader: loads rankings and training set
- FeatureWindowGenerator: creates consecutive feature blocks
- SamplingStrategy: stratified sampling
- ILPRunner: executes PADTAI on each window/sample/seed combination
- ResultsAggregator: consolidates metrics and rules
"""

from .pipeline import Idea2Pipeline
from .data_loader import Idea2DataLoader
from .window_generator import FeatureWindowGenerator
from .sampling import SamplingStrategy
from .ilp_runner import ILPRunner
from .aggregator import ResultsAggregator

__all__ = [
    "Idea2Pipeline",
    "Idea2DataLoader",
    "FeatureWindowGenerator",
    "SamplingStrategy",
    "ILPRunner",
    "ResultsAggregator",
]
