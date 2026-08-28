"""Controlled BSk24 and self-bound CFL equation-of-state experiments."""

from .experiment import (
    Experiment,
    ExperimentPlan,
    ExperimentResult,
    ExperimentSettings,
    load_experiment,
    plan_experiment,
    run_experiment,
    validate_experiment,
)

__version__ = "1.1.0"

__all__ = [
    "Experiment",
    "ExperimentPlan",
    "ExperimentResult",
    "ExperimentSettings",
    "load_experiment",
    "plan_experiment",
    "run_experiment",
    "validate_experiment",
]
