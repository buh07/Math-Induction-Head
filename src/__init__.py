"""
Math Induction Head Experiment Framework

Mechanistic interpretability study on whether induction heads can be forced
to improve arithmetic computation in Llama3-8B via activation patching.
"""

__version__ = "0.1.0"

from . import utils
from . import validation_suite
from . import staged_ablation
from . import multi_metric_measurement
from . import core_experiment
from . import statistical_validation

__all__ = [
    "utils",
    "validation_suite", 
    "staged_ablation",
    "multi_metric_measurement",
    "core_experiment",
    "statistical_validation",
]
