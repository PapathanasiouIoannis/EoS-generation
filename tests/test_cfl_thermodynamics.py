from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad

from eos_generation.cfl import (
    CFLAnalyticEos,
    CFLDomainError,
    CFLWindowedDeformation,
    CFL_PRESSURE_PRIMITIVE_POLICY,
    ENERGY_DENSITY_MAX_MEV_FM3,
    ENERGY_DENSITY_SURFACE_MEV_FM3,
    FROZEN_CFL_PARAMETERS,
    FROZEN_PARAMETER_SET_SHA256,
    build_cfl_baseline,
    build_windowed_eos,
    calculate_windowed_amplitude_bounds,
    make_cfl_eos,
    raw_local_physics_gate,
)
from eos_generation.cfl.baseline import (
    BAG_CONSTANT_NATURAL_MEV4,
    BARYON_CHEMICAL_POTENTIAL_SURFACE_MEV,
    BARYON_DENSITY_SURFACE_FM3,
    HBAR_C_MEV_FM,
    PAIRING_GAP_MEV,
    PHASE1_DISPLAY_REFERENCE_ENERGY_DENSITY_MAX_MEV_FM3,
    PHASE1_DISPLAY_REFERENCE_ENERGY_DENSITY_SURFACE_MEV_FM3,
    PRESSURE_MAX_MEV_FM3,
    QUARK_CHEMICAL_POTENTIAL_MAX_MEV,
    QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV,
    STRANGE_QUARK_MASS_MEV,
    CFLFrozenParameters,
)
from eos_generation.cfl.deformation import (
    gaussian_profile,
    smootherstep_window,
    windowed_gaussian_delta_cs2,
    windowed_gaussian_pressure_primitive,
    windowed_gaussian_shape,
)
from eos_generation.cfl.reconstruction import CFLMechanicalStabilityError


FROZEN_HASH = "3991cb8615d2d29617ccb90c6dc54b23aae64bcc752856d07f17f99abc048307"


@pytest.fixture(scope="module")
def baseline() -> CFLAnalyticEos:
    return make_cfl_eos(513)


@pytest.fixture(scope="module")
def geometry() -> CFLWindowedDeformation:
    return CFLWindowedDeformation(
        case_id="cfl_test_a0",
        amplitude=0.0,
        center_mev_fm3=800.0,
        width_mev_fm3=150.0,
        ramp_width_mev_fm3=100.0,
    )


@pytest.fixture(scope="module")
def bounds(baseline: CFLAnalyticEos, geometry: CFLWindowedDeformation):
    return calculate_windowed_amplitude_bounds(
        center_mev_fm3=geometry.center_mev_fm3,
        width_mev_fm3=geometry.width_mev_fm3,
        ramp_width_mev_fm3=geometry.ramp_width_mev_fm3,
        baseline=baseline,
        discovery_points=513,
    )


@pytest.fixture(scope="module")
def accepted_a0_report(baseline, geometry, bounds):
    report, epsilon, raw_cs2 = raw_local_physics_gate(
        geometry,
        baseline=baseline,
        amplitude_bounds=bounds,
        dense_points=513,
    )
    assert report["status"] == "accepted_raw_local_physics_gate"
    return report, epsilon, raw_cs2


def test_frozen_record_hash_manifest_and_display_reference_roles() -> None:
    record = FROZEN_CFL_PARAMETERS.to_dict()
    assert FROZEN_PARAMETER_SET_SHA256 == FROZEN_HASH
    assert record["parameter_set_sha256"] == FROZEN_HASH
    assert record["formulation_id"] == "cfl_bag_full_ms_delta2_v1"
    json.dumps(record, allow_nan=False)

    manifest_path = (
        Path(__file__).parents[1]
        / "src"
        / "eos_generation"
        / "cfl"
        / "source_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["frozen_parameter_set_sha256"] == FROZEN_HASH
    identity = manifest["binary64_identity_policy"]
    assert identity["authoritative_energy_density_domain_mev_fm3"] == [
        ENERGY_DENSITY_SURFACE_MEV_FM3,
        ENERGY_DENSITY_MAX_MEV_FM3,
    ]
    assert not identity["rounded_display_values_participate_in_identity_or_hash"]
    assert (
        PHASE1_DISPLAY_REFERENCE_ENERGY_DENSITY_SURFACE_MEV_FM3
        != ENERGY_DENSITY_SURFACE_MEV_FM3
    )
    assert (
        PHASE1_DISPLAY_REFERENCE_ENERGY_DENSITY_MAX_MEV_FM3
        != ENERGY_DENSITY_MAX_MEV_FM3
    )
    with pytest.raises(ValueError, match="non-configurable"):
        CFLFrozenParameters(pairing_gap_mev=99.0)


def _independent_full_omega(mu: float) -> float:
    ms = STRANGE_QUARK_MASS_MEV
    gap = PAIRING_GAP_MEV
    nu = 2.0 * mu - math.sqrt(mu**2 + ms**2 / 3.0)
    strange_energy = math.sqrt(nu**2 + ms**2)
    i_massless = nu**4 / 4.0 - mu * nu**3 / 3.0
    i_strange = (
        (
            nu * strange_energy * (2.0 * nu**2 + ms**2)
            - ms**4 * math.asinh(nu / ms)
        )
        / 8.0
        - mu * nu**3 / 3.0
    )
    return (
        6.0 * i_massless / math.pi**2
        + 3.0 * i_strange / math.pi**2
        - 3.0 * gap**2 * mu**2 / math.pi**2
        + BAG_CONSTANT_NATURAL_MEV4
    ) / HBAR_C_MEV_FM**3


def test_full_finite_ms_omega_and_floating_surface_normalization(
    baseline: CFLAnalyticEos,
) -> None:
    raw_surface_pressure = -_independent_full_omega(
        QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV
    )
    for mu in (330.0, 450.0, 575.0):
        expected_pressure = -_independent_full_omega(mu) - raw_surface_pressure
        assert float(baseline.pressure_from_quark_chemical_potential(mu)) == pytest.approx(
            expected_pressure, rel=2.0e-15, abs=2.0e-13
        )
    assert baseline.pressure_from_quark_chemical_potential(
        QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV
    ) == 0.0


def test_analytic_thermodynamic_identities_and_derivatives(
    baseline: CFLAnalyticEos,
) -> None:
    mu = np.asarray(
        [QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV, 350.0, 450.0, 550.0, 600.0]
    )
    pressure = np.asarray(baseline.pressure_from_quark_chemical_potential(mu))
    density = np.asarray(baseline.baryon_density_from_quark_chemical_potential(mu))
    mu_b = np.asarray(
        baseline.baryon_chemical_potential_from_quark_chemical_potential(mu)
    )
    epsilon = np.asarray(
        baseline.energy_density_from_quark_chemical_potential(mu)
    )
    dn_dmu = np.asarray(
        baseline.baryon_density_derivative_from_quark_chemical_potential(mu)
    )
    dp_dmu = np.asarray(
        baseline.pressure_derivative_from_quark_chemical_potential(mu)
    )
    de_dmu = np.asarray(
        baseline.energy_density_derivative_from_quark_chemical_potential(mu)
    )
    cs2 = np.asarray(
        baseline.sound_speed_squared_from_quark_chemical_potential(mu)
    )
    np.testing.assert_array_equal(epsilon, -pressure + mu_b * density)
    np.testing.assert_array_equal(dp_dmu, 3.0 * density)
    np.testing.assert_array_equal(de_dmu, mu_b * dn_dmu)
    np.testing.assert_allclose(cs2, dp_dmu / de_dmu, rtol=2.0e-15, atol=0.0)

    for center in (350.0, 450.0, 550.0):
        step = 1.0e-2
        numeric_dp = (
            baseline.pressure_from_quark_chemical_potential(center + step)
            - baseline.pressure_from_quark_chemical_potential(center - step)
        ) / (2.0 * step)
        numeric_de = (
            baseline.energy_density_from_quark_chemical_potential(center + step)
            - baseline.energy_density_from_quark_chemical_potential(center - step)
        ) / (2.0 * step)
        assert numeric_dp == pytest.approx(
            baseline.pressure_derivative_from_quark_chemical_potential(center),
            rel=2.0e-9,
        )
        assert numeric_de == pytest.approx(
            baseline.energy_density_derivative_from_quark_chemical_potential(center),
            rel=2.0e-9,
        )


def test_surface_stability_domain_and_stellar_metadata(
    baseline: CFLAnalyticEos,
) -> None:
    assert baseline.eps_surf == ENERGY_DENSITY_SURFACE_MEV_FM3
    assert baseline.pressure_min_mev_fm3 == 0.0
    assert baseline.pressure_max_mev_fm3 == PRESSURE_MAX_MEV_FM3
    assert baseline.requires_discontinuity_metadata is True
    assert len(baseline.discontinuities) == 1
    surface = baseline.discontinuities[0]
    assert surface.kind == "surface"
    assert surface.pressure == 0.0
    assert surface.inner_energy_density == baseline.eps_surf
    assert surface.outer_energy_density == 0.0
    assert BARYON_CHEMICAL_POTENTIAL_SURFACE_MEV <= 930.0
    assert (
        STRANGE_QUARK_MASS_MEV**2 / QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV
        < 2.0 * PAIRING_GAP_MEV
    )
    validity = FROZEN_CFL_PARAMETERS.to_dict()["stability_and_validity"]
    assert validity["absolute_stability_passed"] is True
    assert validity["fully_gapped_CFL_passed"] is True
    assert validity["ordinary_nuclei_two_flavor_status"].startswith(
        "external_assumption"
    )
    with pytest.raises(CFLDomainError):
        baseline.energy_density_from_pressure(-1.0e-12)
    with pytest.raises(CFLDomainError):
        baseline.pressure_from_energy_density(
            math.nextafter(ENERGY_DENSITY_MAX_MEV_FM3, math.inf)
        )


def test_monotone_inversions_and_stage_grid_adapter(
    baseline: CFLAnalyticEos,
) -> None:
    mu = np.asarray([QUARK_CHEMICAL_POTENTIAL_SURFACE_MEV, 375.0, 500.0, 600.0])
    epsilon = baseline.energy_density_from_quark_chemical_potential(mu)
    pressure = baseline.pressure_from_quark_chemical_potential(mu)
    np.testing.assert_allclose(
        baseline.quark_chemical_potential_from_energy_density(epsilon),
        mu,
        rtol=0.0,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(
        baseline.quark_chemical_potential_from_pressure(pressure),
        mu,
        rtol=0.0,
        atol=3.0e-12,
    )
    stage = SimpleNamespace(lower_points=33, upper_points=65)
    staged = build_cfl_baseline(stage)
    assert len(staged.epsilon) == 97
    assert staged.settings.source_lower_points == 33
    assert staged.settings.source_upper_points == 65
    assert staged.epsilon[0] == ENERGY_DENSITY_SURFACE_MEV_FM3
    assert staged.epsilon[-1] == ENERGY_DENSITY_MAX_MEV_FM3
    assert staged.eos is staged


def test_analytic_callable_reuses_one_pressure_inversion(
    baseline: CFLAnalyticEos,
    monkeypatch,
) -> None:
    pressure = 0.37 * PRESSURE_MAX_MEV_FM3
    expected = (
        float(baseline.energy_density_from_pressure(pressure)),
        float(baseline.sound_speed_squared_from_pressure(pressure)),
    )
    calls = 0
    original = CFLAnalyticEos.quark_chemical_potential_from_pressure

    def counted(self, value):
        nonlocal calls
        calls += 1
        return original(self, value)

    monkeypatch.setattr(
        CFLAnalyticEos,
        "quark_chemical_potential_from_pressure",
        counted,
    )

    assert baseline(pressure) == expected
    assert calls == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"center_mev_fm3": ENERGY_DENSITY_SURFACE_MEV_FM3},
        {"center_mev_fm3": ENERGY_DENSITY_MAX_MEV_FM3},
        {"width_mev_fm3": 0.0},
        {"ramp_width_mev_fm3": 0.0},
        {"ramp_width_mev_fm3": 1.0e-14},
        {"width_mev_fm3": 1.0e-14},
        {
            "ramp_width_mev_fm3": (
                ENERGY_DENSITY_MAX_MEV_FM3
                - ENERGY_DENSITY_SURFACE_MEV_FM3
                + 1.0
            )
        },
        {"amplitude": math.nan},
    ],
)
def test_deformation_semantics_fail_closed(kwargs: dict[str, float]) -> None:
    parameters = {
        "case_id": "invalid",
        "amplitude": 0.0,
        "center_mev_fm3": 800.0,
        "width_mev_fm3": 150.0,
        "ramp_width_mev_fm3": 100.0,
    }
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        CFLWindowedDeformation(**parameters)


def test_exact_window_geometry_and_analytic_pressure_primitive(
    geometry: CFLWindowedDeformation,
) -> None:
    surface = ENERGY_DENSITY_SURFACE_MEV_FM3
    ramp_end = surface + geometry.ramp_width_mev_fm3
    values = np.asarray([surface - 1.0, surface, surface + 50.0, ramp_end, ramp_end + 1.0])
    expected_mid = 6.0 * 0.5**5 - 15.0 * 0.5**4 + 10.0 * 0.5**3
    np.testing.assert_array_equal(
        smootherstep_window(
            values, ramp_width_mev_fm3=geometry.ramp_width_mev_fm3
        ),
        np.asarray([0.0, 0.0, expected_mid, 1.0, 1.0]),
    )
    assert gaussian_profile(geometry.center_mev_fm3, geometry) == 1.0
    assert windowed_gaussian_shape(surface, geometry) == 0.0
    assert windowed_gaussian_delta_cs2(surface, geometry) == 0.0
    assert windowed_gaussian_pressure_primitive(surface, geometry) == 0.0
    assert windowed_gaussian_pressure_primitive(surface - 20.0, geometry) == 0.0
    with pytest.raises(ValueError, match="representably distinct"):
        smootherstep_window(surface, ramp_width_mev_fm3=1.0e-14)

    nonzero = CFLWindowedDeformation(
        case_id="primitive",
        amplitude=0.05,
        center_mev_fm3=geometry.center_mev_fm3,
        width_mev_fm3=geometry.width_mev_fm3,
        ramp_width_mev_fm3=geometry.ramp_width_mev_fm3,
    )
    for upper in (ramp_end, 800.0, 1200.0):
        independent = nonzero.amplitude * quad(
            lambda epsilon: float(windowed_gaussian_shape(epsilon, nonzero)),
            surface,
            upper,
            epsabs=1.0e-11,
            epsrel=1.0e-12,
        )[0]
        assert windowed_gaussian_pressure_primitive(
            upper, nonzero
        ) == pytest.approx(independent, rel=2.0e-11, abs=3.0e-10)


def _independent_pressure_primitive(
    upper: float,
    deformation: CFLWindowedDeformation,
) -> float:
    """Independent QUADPACK transcription in normalized ramp coordinates."""

    surface = ENERGY_DENSITY_SURFACE_MEV_FM3
    delta = deformation.ramp_width_mev_fm3
    ramp_end = surface + delta
    ramp_fraction = min(1.0, max(0.0, (upper - surface) / delta))

    def ramp_integrand(value: float) -> float:
        window = value**3 * (10.0 + value * (-15.0 + 6.0 * value))
        epsilon = surface + delta * value
        z = (
            epsilon - deformation.center_mev_fm3
        ) / deformation.width_mev_fm3
        return window * math.exp(-0.5 * z * z)

    ramp_area = 0.0
    if ramp_fraction > 0.0:
        ramp_area = delta * quad(
            ramp_integrand,
            0.0,
            ramp_fraction,
            epsabs=2.0e-14,
            epsrel=2.0e-13,
            limit=300,
        )[0]
    tail_area = 0.0
    if upper > ramp_end:
        points = (
            [deformation.center_mev_fm3]
            if ramp_end
            < deformation.center_mev_fm3
            < upper
            else None
        )
        tail_area = quad(
            lambda epsilon: math.exp(
                -0.5
                * (
                    (epsilon - deformation.center_mev_fm3)
                    / deformation.width_mev_fm3
                )
                ** 2
            ),
            ramp_end,
            upper,
            points=points,
            epsabs=2.0e-13,
            epsrel=2.0e-13,
            limit=300,
        )[0]
    return deformation.amplitude * (ramp_area + tail_area)


@pytest.mark.parametrize(
    "ramp_width",
    (1.0e-4, 1.0e-2, 1.0e-1, 1.0, 100.0, 1000.0),
)
def test_pressure_primitive_matches_independent_multiscale_quadrature(
    ramp_width: float,
) -> None:
    deformation = CFLWindowedDeformation(
        case_id=f"multiscale_{ramp_width}",
        amplitude=0.01,
        center_mev_fm3=800.0,
        width_mev_fm3=100.0,
        ramp_width_mev_fm3=ramp_width,
    )
    surface = ENERGY_DENSITY_SURFACE_MEV_FM3
    ramp_end = surface + ramp_width
    for upper in (
        surface + 0.37 * ramp_width,
        ramp_end,
        min(ENERGY_DENSITY_MAX_MEV_FM3, max(ramp_end, 1000.0)),
    ):
        observed = windowed_gaussian_pressure_primitive(upper, deformation)
        expected = _independent_pressure_primitive(upper, deformation)
        assert observed == pytest.approx(
            expected,
            rel=3.0e-11,
            abs=2.0e-13,
        )


@pytest.mark.parametrize("ramp_width", (1.0e-2, 1.0, 100.0))
def test_pressure_primitive_independent_finite_difference_derivative(
    ramp_width: float,
) -> None:
    deformation = CFLWindowedDeformation(
        case_id=f"derivative_{ramp_width}",
        amplitude=0.01,
        center_mev_fm3=800.0,
        width_mev_fm3=100.0,
        ramp_width_mev_fm3=ramp_width,
    )
    surface = ENERGY_DENSITY_SURFACE_MEV_FM3
    epsilon = surface + 0.53 * ramp_width
    step = max(1.0e-8, ramp_width * 1.0e-5)
    derivative = (
        windowed_gaussian_pressure_primitive(epsilon + step, deformation)
        - windowed_gaussian_pressure_primitive(epsilon - step, deformation)
    ) / (2.0 * step)
    expected = windowed_gaussian_delta_cs2(epsilon, deformation)
    assert derivative == pytest.approx(expected, rel=2.0e-6, abs=2.0e-12)


def test_small_valid_ramp_raw_gate_uses_stable_pressure_primitive(
    baseline: CFLAnalyticEos,
) -> None:
    deformation = CFLWindowedDeformation(
        case_id="small_valid_ramp",
        amplitude=0.01,
        center_mev_fm3=800.0,
        width_mev_fm3=100.0,
        ramp_width_mev_fm3=0.1,
    )
    report, _, _ = raw_local_physics_gate(
        deformation,
        baseline=baseline,
        dense_points=257,
    )
    assert report["status"] == "accepted_raw_local_physics_gate"
    assert report["nonnegative_pressure_including_zero_surface"] is True
    assert report["pressure_primitive_policy"] == (
        CFL_PRESSURE_PRIMITIVE_POLICY
    )


def test_continuous_bounds_and_full_domain_a0_gate(
    baseline, geometry, bounds, monkeypatch
) -> None:
    assert bounds.amplitude_min < 0.0 < bounds.amplitude_max
    assert bounds.contains(0.0)
    lower_cs2 = baseline.sound_speed_squared_from_energy_density(
        bounds.lower_limiting_epsilon_mev_fm3
    )
    upper_cs2 = baseline.sound_speed_squared_from_energy_density(
        bounds.upper_limiting_epsilon_mev_fm3
    )
    assert lower_cs2 + bounds.amplitude_min * bounds.lower_limiting_shape == pytest.approx(
        0.0, abs=3.0e-15
    )
    assert upper_cs2 + bounds.amplitude_max * bounds.upper_limiting_shape == pytest.approx(
        1.0, abs=3.0e-15
    )
    sampled_baseline_cs2: list[np.ndarray] = []
    original_cs2 = CFLAnalyticEos.sound_speed_squared_from_quark_chemical_potential

    def capture_sampled_baseline(self, value):
        result = original_cs2(self, value)
        sampled = np.asarray(result, dtype=float)
        if sampled.ndim == 1 and sampled.size > 1:
            sampled_baseline_cs2.append(sampled.copy())
        return result

    monkeypatch.setattr(
        CFLAnalyticEos,
        "sound_speed_squared_from_quark_chemical_potential",
        capture_sampled_baseline,
    )
    report, epsilon, raw_cs2 = raw_local_physics_gate(
        geometry,
        baseline=baseline,
        amplitude_bounds=bounds,
        dense_points=513,
    )
    assert report["full_declared_domain_passed"] is True
    assert report["surface"]["preserved_exactly"] is True
    assert report["clipping_clamping_smoothing_posthoc_repair"] == "none"
    assert epsilon[0] == ENERGY_DENSITY_SURFACE_MEV_FM3
    assert epsilon[-1] == ENERGY_DENSITY_MAX_MEV_FM3
    # Capture the baseline values sampled on the gate's native chemical-
    # potential grid.  This checks the actual zero-amplitude invariant exactly,
    # without introducing a platform-sensitive epsilon -> mu Brent round trip.
    assert len(sampled_baseline_cs2) == 1
    np.testing.assert_array_equal(raw_cs2, sampled_baseline_cs2[0])
    np.testing.assert_array_equal(
        windowed_gaussian_delta_cs2(epsilon, geometry),
        np.zeros_like(epsilon),
    )
    json.dumps(report, allow_nan=False)


def test_bounds_reject_open_lower_and_above_closed_upper(baseline, geometry, bounds) -> None:
    lower = CFLWindowedDeformation(
        case_id="lower_bound",
        amplitude=bounds.amplitude_min,
        center_mev_fm3=geometry.center_mev_fm3,
        width_mev_fm3=geometry.width_mev_fm3,
        ramp_width_mev_fm3=geometry.ramp_width_mev_fm3,
    )
    lower_report, _, _ = raw_local_physics_gate(
        lower,
        baseline=baseline,
        amplitude_bounds=bounds,
        dense_points=513,
    )
    assert lower_report["status"] == "rejected_raw_local_physics_gate"
    assert lower_report["first_failure"]["reason"] == (
        "mechanical_stability_nonpositive_cs2"
    )
    assert lower_report["first_failure"]["raw_cs2"] == pytest.approx(0.0, abs=3.0e-15)

    upper = CFLWindowedDeformation(
        case_id="above_upper_bound",
        amplitude=math.nextafter(bounds.amplitude_max, math.inf),
        center_mev_fm3=geometry.center_mev_fm3,
        width_mev_fm3=geometry.width_mev_fm3,
        ramp_width_mev_fm3=geometry.ramp_width_mev_fm3,
    )
    upper_report, _, _ = raw_local_physics_gate(
        upper,
        baseline=baseline,
        amplitude_bounds=bounds,
        dense_points=513,
    )
    assert upper_report["status"] == "rejected_raw_local_physics_gate"
    assert upper_report["first_failure"]["reason"] == "causality_superluminal_cs2"
    assert upper_report["first_failure"]["raw_cs2"] >= 1.0
    assert upper_report["first_failure"]["amplitude_interval_violation"] == (
        "above_closed_upper_bound"
    )

    closed = CFLWindowedDeformation(
        case_id="closed_upper_bound",
        amplitude=bounds.amplitude_max,
        center_mev_fm3=geometry.center_mev_fm3,
        width_mev_fm3=geometry.width_mev_fm3,
        ramp_width_mev_fm3=geometry.ramp_width_mev_fm3,
    )
    closed_report, _, _ = raw_local_physics_gate(
        closed,
        baseline=baseline,
        amplitude_bounds=bounds,
        dense_points=513,
    )
    assert closed_report["status"] == "accepted_raw_local_physics_gate"
    assert closed_report["amplitude_interval_passed"] is True


def test_finite_but_overflowing_amplitude_is_retained_as_strict_json_rejection(
    baseline, geometry, bounds
) -> None:
    deformation = CFLWindowedDeformation(
        case_id="finite_overflow",
        amplitude=1.0e308,
        center_mev_fm3=geometry.center_mev_fm3,
        width_mev_fm3=geometry.width_mev_fm3,
        ramp_width_mev_fm3=geometry.ramp_width_mev_fm3,
    )
    report, _epsilon, raw_cs2 = raw_local_physics_gate(
        deformation,
        baseline=baseline,
        amplitude_bounds=bounds,
        dense_points=257,
    )
    assert report["status"] == "rejected_raw_local_physics_gate"
    assert report["first_failure"]["reason"] == "nonfinite_raw_state"
    assert np.any(~np.isfinite(raw_cs2)) or (
        report["first_failure"]["pressure_classification"] != "finite"
    )
    json.dumps(report, allow_nan=False)


def test_reconstruction_requires_untampered_authoritative_report(
    geometry, accepted_a0_report
) -> None:
    report, _, _ = accepted_a0_report
    with pytest.raises(ValueError, match="requires an authoritative"):
        build_windowed_eos(geometry, grid_points=513)
    tampered = dict(report)
    tampered["raw_minimum_cs2"] = 0.5
    with pytest.raises(ValueError, match="hash does not match"):
        build_windowed_eos(geometry, raw_gate_report=tampered, grid_points=513)
    rejected = dict(report)
    rejected["status"] = "rejected_raw_local_physics_gate"
    rejected.pop("report_sha256")
    from eos_generation.cfl.deformation import _canonical_sha256

    rejected["report_sha256"] = _canonical_sha256(rejected)
    with pytest.raises(CFLMechanicalStabilityError):
        build_windowed_eos(geometry, raw_gate_report=rejected, grid_points=513)


def test_zero_amplitude_analytic_identity_and_case_specific_surface(
    baseline, geometry, accepted_a0_report
) -> None:
    report, _, _ = accepted_a0_report
    generated = build_windowed_eos(
        geometry,
        baseline=baseline,
        raw_gate_report=report,
        grid_points=513,
    )
    np.testing.assert_array_equal(generated.epsilon, baseline.epsilon)
    np.testing.assert_array_equal(generated.pressure, baseline.pressure)
    np.testing.assert_array_equal(generated.cs2, baseline.cs2)
    np.testing.assert_array_equal(
        generated.baryon_density, baseline.baryon_density
    )
    np.testing.assert_array_equal(
        generated.baryon_chemical_potential, baseline.chemical_potential
    )
    sample_epsilon = np.asarray(
        [ENERGY_DENSITY_SURFACE_MEV_FM3, 500.0, 1000.0, ENERGY_DENSITY_MAX_MEV_FM3]
    )
    np.testing.assert_array_equal(
        generated.pressure_from_energy_density(sample_epsilon),
        baseline.pressure_from_energy_density(sample_epsilon),
    )
    np.testing.assert_array_equal(
        generated.cs2_from_energy_density(sample_epsilon),
        baseline.cs2_from_energy_density(sample_epsilon),
    )
    sample_pressure = np.asarray([0.0, 100.0, 800.0, PRESSURE_MAX_MEV_FM3])
    np.testing.assert_array_equal(
        generated.energy_density_from_pressure(sample_pressure),
        baseline.energy_density_from_pressure(sample_pressure),
    )
    assert generated.eps_surf == baseline.eps_surf
    assert len(generated.discontinuities) == 1
    assert geometry.case_sha256 in generated.discontinuities[0].provenance
    assert generated.to_dict()["zero_amplitude_analytic_identity"] is True


def test_nonzero_reconstruction_first_law_surface_and_inverse(
    baseline, geometry, bounds
) -> None:
    deformation = CFLWindowedDeformation(
        case_id="positive_case",
        amplitude=0.05,
        center_mev_fm3=geometry.center_mev_fm3,
        width_mev_fm3=geometry.width_mev_fm3,
        ramp_width_mev_fm3=geometry.ramp_width_mev_fm3,
    )
    assert bounds.contains(deformation.amplitude)
    report, _, _ = raw_local_physics_gate(
        deformation,
        baseline=baseline,
        amplitude_bounds=bounds,
        dense_points=513,
    )
    assert report["status"] == "accepted_raw_local_physics_gate"
    generated = build_windowed_eos(
        deformation,
        baseline=baseline,
        raw_gate_report=report,
        grid_points=1025,
    )
    assert generated.epsilon[0] == ENERGY_DENSITY_SURFACE_MEV_FM3
    assert generated.pressure[0] == 0.0
    assert generated.baryon_density[0] == BARYON_DENSITY_SURFACE_FM3
    assert (
        generated.baryon_chemical_potential[0]
        == BARYON_CHEMICAL_POTENTIAL_SURFACE_MEV
    )
    assert np.all(np.diff(generated.pressure) > 0.0)
    assert np.all((generated.cs2 > 0.0) & (generated.cs2 <= 1.0))
    np.testing.assert_allclose(
        generated.pressure,
        generated.baryon_density * generated.baryon_chemical_potential
        - generated.epsilon,
        rtol=0.0,
        atol=8.0e-13,
    )
    interior = generated.epsilon[20:-20:100]
    forward_pressure = generated.pressure_from_energy_density(interior)
    inverse_epsilon = generated.energy_density_from_pressure(forward_pressure)
    np.testing.assert_allclose(inverse_epsilon, interior, rtol=2.0e-9, atol=2.0e-7)
    query = 0.5 * (generated.epsilon[40:-40:137] + generated.epsilon[41:-39:137])
    query_pressure = generated.pressure_from_energy_density(query)
    query_density = generated.baryon_density_from_energy_density(query)
    query_mu_b = generated.baryon_chemical_potential_from_energy_density(query)
    np.testing.assert_array_equal(
        query_mu_b,
        (query + query_pressure) / query_density,
    )
    np.testing.assert_allclose(
        query_pressure,
        query_density * query_mu_b - query,
        rtol=0.0,
        atol=8.0e-13,
    )
    assert len(generated.discontinuities) == 1
    assert generated.discontinuities[0].inner_energy_density == generated.eps_surf
    assert deformation.case_sha256 in generated.discontinuities[0].provenance
