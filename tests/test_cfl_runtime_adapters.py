from __future__ import annotations

import numpy as np
import pytest

from eos_generation import ExperimentSettings, plan_experiment
from eos_generation._internal.config import DEFAULT_CONFIG
from eos_generation._internal.cfl_thermodynamics import (
    _cfl_a0_identity_table,
    _cfl_deformations,
    _cfl_thermodynamic_convergence,
    _cfl_thermodynamic_profile_frame,
    _cfl_thermodynamic_residual_frame,
)
from eos_generation._internal.execution import (
    _assert_cfl_a0_gate_invariant,
    _logical_outcome_ids,
)
from eos_generation.cfl import (
    build_cfl_baseline,
    build_windowed_eos,
    raw_local_physics_gate,
)
from eos_generation.stellar._tov_sequence import _sequence_process_worker_is_safe


def _owner_child():
    settings = ExperimentSettings.from_values(
        matter_model="cfl",
        epsilon_match="surface",
        amplitudes=(0.0, 0.02),
        center=800.0,
        width=150.0,
        ramp_width=100.0,
        calculation="thermodynamics",
        precision="quick",
    )
    return plan_experiment(settings).child_plans[0]


def test_cfl_packet_adapters_retain_physical_identity_and_a0_arrays() -> None:
    child = _owner_child()
    deformations = _cfl_deformations(child)
    zero = [item for item in deformations.values() if item.amplitude == 0.0]
    assert len(zero) == 1
    assert zero[0].case_id.startswith("cfl_baseline_")

    baseline = build_cfl_baseline(
        child.config.thermodynamic_stages[0].grid_settings()
    )
    report, _, _ = raw_local_physics_gate(
        zero[0], baseline=baseline, dense_points=257
    )
    generated = build_windowed_eos(
        zero[0],
        baseline=baseline,
        raw_gate_report=report,
        grid_points=baseline.settings.points,
    )
    np.testing.assert_array_equal(generated.epsilon, baseline.epsilon)
    np.testing.assert_array_equal(generated.pressure, baseline.pressure)
    assert baseline.allow_parallel_tov_sequence is True
    assert generated.allow_parallel_tov_sequence is True
    for eos in (baseline, generated):
        assert _sequence_process_worker_is_safe(
            eos,
            DEFAULT_CONFIG.tov,
            1.0e-8,
            1.0e-10,
            True,
        )

    profile = _cfl_thermodynamic_profile_frame(
        baseline, {zero[0].case_id: generated}
    )
    assert set(profile["matter_model"]) == {"cfl"}
    assert set(profile["case_id"]) == {"direct", zero[0].case_id}
    surface_rows = profile.loc[
        profile["epsilon_mev_fm3"].eq(baseline.epsilon[0])
    ]
    assert surface_rows["pressure_mev_fm3"].eq(0.0).all()
    assert surface_rows["pressure_relative_to_direct"].isna().all()

    identity, table = _cfl_a0_identity_table(
        baseline=baseline,
        generated={zero[0].case_id: generated},
        config=child.config,
        sequences=None,
        fixed=None,
    )
    assert identity["status"] == "pass"
    assert table["array_equal"].all()
    assert table["maximum_absolute_residual"].eq(0.0).all()


def test_cfl_residual_and_convergence_evidence_are_model_specific() -> None:
    child = _owner_child()
    deformation = next(
        item
        for item in _cfl_deformations(child).values()
        if item.amplitude == 0.0
    )
    baseline = build_cfl_baseline(
        child.config.thermodynamic_stages[0].grid_settings()
    )
    report, _, _ = raw_local_physics_gate(
        deformation, baseline=baseline, dense_points=257
    )
    generated = build_windowed_eos(
        deformation,
        baseline=baseline,
        raw_gate_report=report,
        grid_points=baseline.settings.points,
    )
    residuals = _cfl_thermodynamic_residual_frame(
        {deformation.case_id: generated}
    )
    assert set(residuals["matter_model"]) == {"cfl"}
    for name in (
        "r_p_independent_normalized",
        "r_mu_independent_normalized",
        "first_law_normalized",
        "r_c",
    ):
        assert name in residuals
        assert np.isfinite(residuals[name]).all()
        np.testing.assert_array_equal(
            residuals[name].to_numpy(dtype=float),
            generated.residuals[name],
        )

    convergence = _cfl_thermodynamic_convergence(
        {"smoke": {deformation.case_id: generated}}
    )
    assert convergence["matter_model"] == "cfl"
    assert convergence["status"] == "insufficient_stages"


def test_runtime_dispatches_cfl_through_shared_lifecycle(monkeypatch) -> None:
    from eos_generation._internal import runtime

    child = _owner_child()
    sentinel = object()
    captured = {}

    def fake_run(config, *, callbacks):
        captured["config"] = config
        captured["callbacks"] = callbacks
        return sentinel

    monkeypatch.setattr(runtime, "run_cfl_trial", fake_run)
    assert runtime.execute_trial(child.config) is sentinel
    assert captured["config"] is child.config
    callbacks = captured["callbacks"]
    assert callbacks.resolve_deformations is _cfl_deformations
    assert callbacks.a0_identity_table is _cfl_a0_identity_table


def test_runtime_maps_physical_a0_outcome_to_its_logical_case() -> None:
    child = _owner_child()
    zero_row = child.case_table.loc[child.case_table["amplitude"].eq(0.0)].iloc[0]
    accepted, rejected = _logical_outcome_ids(
        child,
        accepted_physical_ids=[str(zero_row["physical_case_id"])],
        rejected_physical_ids=[
            str(
                child.case_table.loc[
                    child.case_table["amplitude"].ne(0.0), "physical_case_id"
                ].iloc[0]
            )
        ],
        is_cfl=True,
    )
    assert accepted == [str(zero_row["case_id"])]
    assert rejected == [
        str(
            child.case_table.loc[
                child.case_table["amplitude"].ne(0.0), "case_id"
            ].iloc[0]
        )
    ]


def test_cfl_owner_rejects_missing_a0_before_reconstruction() -> None:
    child = _owner_child()
    deformations = _cfl_deformations(child)
    zero_id = next(
        case_id
        for case_id, deformation in deformations.items()
        if deformation.amplitude == 0.0
    )
    with pytest.raises(
        RuntimeError,
        match="mandatory owned CFL A=0 control failed",
    ):
        _assert_cfl_a0_gate_invariant(
            child.config,
            deformations,
            {zero_id: {"first_failure": "synthetic_raw_failure"}},
            [case_id for case_id in deformations if case_id != zero_id],
        )


def test_cfl_identity_report_fails_when_owner_has_no_generated_a0() -> None:
    child = _owner_child()
    baseline = build_cfl_baseline(
        child.config.thermodynamic_stages[0].grid_settings()
    )
    identity, table = _cfl_a0_identity_table(
        baseline=baseline,
        generated={},
        config=child.config,
        sequences=None,
        fixed=None,
    )
    assert identity["status"] == "fail"
    assert identity["stellar_identity_status"] == (
        "fail_missing_or_rejected_owned_a0_case"
    )
    assert table.empty


def test_cfl_completion_uses_saved_direct_rows_for_the_physical_zero_alias():
    import pandas as pd
    from eos_generation._internal.lifecycle import _completed_stellar_case_ids
    from eos_generation.stellar.tov import LAMBDA_FRAMEWORK_CAPABILITY

    settings = ExperimentSettings.from_values(
        matter_model="cfl", epsilon_match="surface", amplitudes=(0.0,),
        center=800.0, width=150.0, ramp_width=100.0, calculation="stellar", precision="quick",
    )
    config = plan_experiment(settings).child_plans[0].config
    zero_id = config.to_dict()["zero_amplitude_physical_case_id"]
    stage = config.tov_stages[0]
    sequence = pd.DataFrame([dict(case_id="direct", stage=stage.name, calculation_status="success", k2=0.1, Lambda=100.0,
                                  tidal_status=LAMBDA_FRAMEWORK_CAPABILITY)] * stage.sequence_points)
    fixed = pd.DataFrame([dict(case_id="direct", stage=stage.name, status="bracketed_and_solved", k2=0.1,
                               lambda_dimensionless=100.0, tidal_status=LAMBDA_FRAMEWORK_CAPABILITY)])
    assert _completed_stellar_case_ids(sequence, fixed, config, accepted_case_ids=[zero_id]) == {zero_id}
    sequence.loc[0, "tidal_status"] = "failed_closed"
    assert _completed_stellar_case_ids(sequence, fixed, config, accepted_case_ids=[zero_id]) == set()
