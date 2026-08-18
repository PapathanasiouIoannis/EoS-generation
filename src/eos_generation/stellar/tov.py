"""Sole production authority for TOV backgrounds and individual-star tides.

The background equations retain the repository's historical units. Tidal
integration is branch-segmented and applies explicitly declared finite-pressure
and bare-surface energy-density matching conditions before evaluating ``k2``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, ClassVar

from eos_generation._internal.config import DEFAULT_CONFIG, TovConfig
from eos_generation.stellar._tov_algebra import (
    _love_number_k2_decimal,
    _require_finite,
    _tov_equations_with_limit,
    dimensionless_lambda,
    love_number_k2,
    surface_event,
    taylor_expansion,
    tidal_algebra,
    tidal_jump_delta_y,
    tov_equations,
    tov_rhs,
)
from eos_generation.stellar._tov_integration import (
    _BackgroundSegment,
    _applicable_discontinuities,
    _assert_tidal_radius_on_segment,
    _background_only_lambda_diagnostic,
    _background_rhs,
    _bounded_tidal_first_step,
    _branch_pressure,
    _evaluate_branch,
    _failed_lambda_diagnostic,
    _integrate_background,
    _integrate_tidal,
    _pressure_event,
    _profile_from_segments,
    _resolved_discontinuities,
    _tidal_rhs_on_segment,
    _tidal_segment_bounds,
    _validate_declared_branch_values,
    solve_star,
)
from eos_generation.stellar._tov_maximum import (
    refine_maximum_mass_from_sequence,
    resolve_maximum_mass,
)
from eos_generation.stellar._tov_sequence import (
    _PRODUCTION_SOLVE_STAR,
    _SEQUENCE_WORKER_STATE,
    _automatic_sequence_worker_count,
    _build_sequence_evidence,
    _initialize_sequence_worker,
    _sampled_mass_secants,
    _sequence_process_worker_is_safe,
    _solve_sequence_pressure_worker,
    _successful_pressure_ordering,
    _tidal_diagnostic_rows,
    solve_sequence,
)
from eos_generation.stellar._tov_types import (
    DENOMINATOR_FLOOR,
    LAMBDA_FRAMEWORK_CAPABILITY,
    LAMBDA_SCIENTIFIC_STATUS,
    TIDAL_CORRECTION_STATUS,
    TIDAL_CORRECTION_VERSION,
    TIDAL_JUMP_FORMULA,
    TIDAL_JUMP_SOURCES,
    TIDAL_NOT_REQUESTED_STATUS,
    TOV_SEQUENCE_EVIDENCE_VERSION,
    TOV_SEQUENCE_FIELDS,
    TOV_TIDAL_DIAGNOSTIC_FIELDS,
    AppliedTidalJump,
    TidalAlgebraResult,
    TovConvergenceError,
    TovFailureDetail,
    TovLambdaDiagnostic,
    TovMassSecantEvidence,
    TovMaximumMassResult,
    TovSequenceEvidence,
    TovStarResult,
    _ABSOLUTE_P_MAX_FALLBACK,
    _A_CONV,
    _BUCHDAHL_LIMIT,
    _DEFAULT_TOV,
    _G_CONV,
    _MAXIMUM_AUTOMATIC_SEQUENCE_WORKERS,
    _MIN_MASS_CUTOFF,
    _MIN_RADIUS_CUTOFF,
    _OUTER_PARALLEL_WORKER_ENV,
    _P_MIN_SAFE,
    _R_MAX,
    _R_MIN,
    _TIDAL_SEGMENT_BOUNDARY_ULPS,
    _TOV_SINGULARITY_LIMIT,
    _evidence_finite_float,
    _freeze_profiles,
    _freeze_rows,
    _optional_finite,
    _rows_equal,
    _rows_to_dicts,
)
from eos_generation.stellar.discontinuities import (
    EOS_DISCONTINUITY_CONTRACT_VERSION,
    EosDiscontinuity,
    validate_discontinuity_sequence,
)


logger = logging.getLogger(__name__)


__all__ = [
    "AppliedTidalJump",
    "DENOMINATOR_FLOOR",
    "LAMBDA_FRAMEWORK_CAPABILITY",
    "LAMBDA_SCIENTIFIC_STATUS",
    "TIDAL_CORRECTION_STATUS",
    "TIDAL_CORRECTION_VERSION",
    "TIDAL_JUMP_FORMULA",
    "TIDAL_JUMP_SOURCES",
    "TIDAL_NOT_REQUESTED_STATUS",
    "TOV_SEQUENCE_EVIDENCE_VERSION",
    "TOV_SEQUENCE_FIELDS",
    "TOV_TIDAL_DIAGNOSTIC_FIELDS",
    "TidalAlgebraResult",
    "TovConvergenceError",
    "TovFailureDetail",
    "TovLambdaDiagnostic",
    "TovMassSecantEvidence",
    "TovMaximumMassResult",
    "TovSequenceEvidence",
    "TovStarResult",
    "dimensionless_lambda",
    "love_number_k2",
    "refine_maximum_mass_from_sequence",
    "resolve_maximum_mass",
    "solve_star",
    "solve_sequence",
    "surface_event",
    "taylor_expansion",
    "tidal_algebra",
    "tidal_jump_delta_y",
    "tov_equations",
    "tov_rhs",
]
