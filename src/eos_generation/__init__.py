"""Controlled analytical BSk24 / BSk25 equation-of-state experiments."""

from __future__ import annotations


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


__version__ = "2.0.0"

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
