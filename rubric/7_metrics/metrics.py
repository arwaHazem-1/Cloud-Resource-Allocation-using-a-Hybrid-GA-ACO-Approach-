"""Fitness/objectives and evaluation metrics.

This is a rubric-facing façade over `fitness/evaluator.py`, kept here so the
project layout mirrors the evaluation categories.
"""

from fitness.evaluator import (  # noqa: F401
    FitnessConfig,
    evaluate,
    evaluate_components,
    evaluate_metrics,
)

