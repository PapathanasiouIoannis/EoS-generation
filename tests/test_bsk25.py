"""Independent paper references and model-selection workflow contracts."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from eos_generation import (
    ExperimentSettings, plan_experiment, run_experiment, load_experiment,
    validate_experiment,
)
from eos_generation import baseline, thermodynamics, stellar
from eos_generation.settings import _PRECISIONS
from eos_generation.storage import sha256

FIXTURE = Path(__file__).parent / "fixtures/bsk25_contract_v1"
REFERENCE = json.loads((FIXTURE / "reference.json").read_text(encoding="utf-8"))


def test_paper_pressure_and_independent_numerical_derivative():
    eos = baseline.make_baseline_eos("bsk25")
    rows = REFERENCE["pressure_rows"]
    rho = np.array([r["rho_g_cm3"] for r in rows])
    np.testing.assert_allclose(
        eos.published_fit_pressure_from_mass_density(rho),
        [r["pressure_mev_fm3"] for r in rows], rtol=5e-13, atol=0,
    )
    np.testing.assert_allclose(
        eos.published_fit_sound_speed_squared_from_mass_density(rho),
        [r["cs2"] for r in rows], rtol=5e-13, atol=5e-14,
    )
    assert rows[-1]["cs2"] > 1
    with pytest.raises(baseline.BSk24DomainError):
        eos.pressure_from_mass_density(rho[-1])
    with pytest.raises(baseline.BSk24DomainError):
        eos.published_fit_pressure_from_mass_density(1.001e16)


def test_paper_c1_and_model_specific_anchor_and_boundaries():
    eos = baseline.make_baseline_eos("bsk25")
    for row in REFERENCE["c1_rows"]:
        epsilon = eos._mass_density_from_baryon_density(row["n_b_fm3"]) / baseline.MEV_FM3_TO_MASS_DENSITY_G_CM3
        assert epsilon == pytest.approx(row["epsilon_mev_fm3"], rel=5e-14)
    anchor = thermodynamics.approved_anchor_state(eos)
    assert anchor.baryon_density_fm3 == 0.16
    assert anchor.energy_density_mev_fm3 == pytest.approx(REFERENCE["standard_anchor_epsilon_mev_fm3"], rel=5e-14)
    definition = eos.definition
    assert definition.core_entry_epsilon_mev_fm3 == REFERENCE["core_entry"]["epsilon_mev_fm3"]
    assert definition.core_entry_baryon_density_fm3 == REFERENCE["core_entry"]["n_b_fm3"]
    assert definition.muon_onset_baryon_density_fm3 == REFERENCE["muon_onset_n_b_fm3"]
    assert definition.outer_inner_transition_epsilon_mev_fm3 == REFERENCE["outer_inner_boundary"]["epsilon_mev_fm3"]
    assert definition.energy_coefficients[7] == 2.54
    assert definition.energy_low_exponent == 7 / 6
    provenance = eos.provenance()
    assert "BSK25" in provenance["model_name"]
    assert provenance["source_discrepancies"]["C1_p8"]["paper"] == 2.54
    assert provenance["source_checksums_sha256"]["compose_eos_zip"] == REFERENCE["source_hashes"]["eos.zip"]


def test_causal_domain_and_inverse_units():
    eos = baseline.make_baseline_eos("bsk25")
    rho = np.geomspace(1e6, 3.81e15, 4097)
    pressure = np.asarray(eos.pressure_from_mass_density(rho))
    cs2 = np.asarray(eos.sound_speed_squared_from_mass_density(rho))
    assert np.all(np.diff(pressure) > 0)
    assert np.all((cs2 > 0) & (cs2 <= 1))
    np.testing.assert_allclose(eos.mass_density_from_pressure(pressure[::256]), rho[::256], rtol=5e-12)
    np.testing.assert_allclose(eos.energy_density_from_pressure(pressure[::256]), rho[::256] / baseline.MEV_FM3_TO_MASS_DENSITY_G_CM3, rtol=5e-12)
    epsilon_max = eos.energy_density_max_causal_mev_fm3
    assert float(eos.mass_density_from_energy_density(epsilon_max)) == 3.81e15
    with pytest.raises(baseline.BSk24DomainError):
        eos.pressure_from_energy_density(epsilon_max * 1.000001)


def test_model_identity_and_legacy_bsk24_case_ids():
    legacy = ExperimentSettings.from_values()
    assert "matter_model" not in legacy.to_dict()
    assert legacy.to_dict() == ExperimentSettings.from_values(matter_model="bsk24").to_dict()
    old = plan_experiment(legacy).case_table
    assert old.case_id.tolist() == ["dp40_a0_88dcc71ce4", "dp40_ap0p01_9520f70d33"]
    assert old.physical_case_id.tolist() == ["bsk24_baseline_1b29e04ebe2b0ee7", "dp40_ap0p01_9520f70d33"]
    new = ExperimentSettings.from_values(matter_model="bsk25")
    assert new.to_dict()["matter_model"] == "bsk25"
    assert ExperimentSettings.from_dict(new.to_dict()) == new
    assert new.deterministic_hash() != legacy.deterministic_hash()
    table = plan_experiment(new).case_table
    assert set(table.physical_case_id).isdisjoint(old.physical_case_id)
    assert table.physical_case_id.str.startswith("bsk25_").all()


@pytest.mark.parametrize("precision", _PRECISIONS)
@pytest.mark.parametrize("calculation,diagnostics", [("thermodynamics", "off"), ("stellar", "off"), ("stellar", "on")])
def test_all_profiles_plan_bsk25_without_solving_or_writing(tmp_path, precision, calculation, diagnostics):
    settings = ExperimentSettings.from_values(
        matter_model="bsk25", precision=precision, calculation=calculation,
        diagnostics=diagnostics, center=[200, 250], width=[50, 100],
        ramp_width=[40, 60], amplitudes=[-.01, .01],
        observables=["sequence", "fixed_mass", "maximum_mass"] if calculation == "stellar" else [],
    )
    with (
        patch.object(thermodynamics, "build_consistent_baseline", side_effect=AssertionError("solver")),
        patch.object(stellar, "solve_star", side_effect=AssertionError("solver")),
    ):
        first = plan_experiment(settings, output_path=tmp_path / "runs/study")
        second = plan_experiment(settings, output_path=tmp_path / "runs/study")
    assert not list(tmp_path.iterdir())
    assert first.to_dict() == second.to_dict()
    zero = first.case_table[first.case_table.amplitude.eq(0)]
    assert len(zero) == 8 and zero.physical_case_id.nunique() == 1
    assert zero.planned_for_execution.sum() == 1
    assert set(first.case_table.epsilon_match_mev_fm3) == {REFERENCE["standard_anchor_epsilon_mev_fm3"]}


def test_numeric_anchor_validation_is_model_specific():
    plan_experiment(ExperimentSettings.from_values(epsilon_match=80))
    with pytest.raises(ValueError, match="BSK25.*standard"):
        plan_experiment(ExperimentSettings.from_values(matter_model="bsk25", epsilon_match=80))
    plan = plan_experiment(ExperimentSettings.from_values(matter_model="bsk25", epsilon_match=90))
    assert set(plan.case_table.epsilon_match_mev_fm3) == {90}


@pytest.fixture(scope="module")
def bsk25_result(tmp_path_factory):
    destination = tmp_path_factory.mktemp("bsk25") / "runs/study"
    settings = ExperimentSettings.from_values(matter_model="bsk25", amplitudes=[0, -.01, .01, -2])
    with patch.object(stellar, "_run_stellar", side_effect=AssertionError("stellar work forbidden")):
        return run_experiment(plan_experiment(settings, output_path=destination), execute=True)


def test_new_model_saved_contract_zero_identity_and_rejection(bsk25_result):
    result = bsk25_result
    assert result.settings.matter_model == "bsk25"
    assert result.metadata["settings"]["matter_model"] == "bsk25"
    assert "BSK25" in result.metadata["baseline"]["provenance"]["model_name"]
    cases = result.case_table
    assert cases.loc[cases.amplitude.eq(-2), "status"].eq("rejected").all()
    rejected_ids = set(cases.loc[cases.status.eq("rejected"), "physical_case_id"])
    profiles = result.thermodynamic_profiles
    assert set(profiles.case_id).isdisjoint(rejected_ids)
    zero_id = cases.loc[cases.amplitude.eq(0), "physical_case_id"].iloc[0]
    zero = profiles[profiles.case_id.eq(zero_id)]
    eos = baseline.make_baseline_eos("bsk25")
    rho = eos.mass_density_from_energy_density(zero.epsilon_mev_fm3.to_numpy())
    np.testing.assert_allclose(zero.pressure_mev_fm3, eos.pressure_from_mass_density(rho), rtol=5e-13, atol=0)
    np.testing.assert_allclose(zero.cs2, eos.sound_speed_squared_from_mass_density(rho), rtol=5e-13, atol=5e-14)
    np.testing.assert_allclose(zero.effective_baryon_enthalpy_mev * zero.baryon_density_fm3,
                               zero.epsilon_mev_fm3 + zero.pressure_mev_fm3, rtol=5e-13)
    assert validate_experiment(result.experiment_path)["passed"]
    assert load_experiment(result.experiment_path).settings == result.settings


def test_saved_bsk25_plot_label_and_read_only_science(bsk25_result):
    from eos_generation import plotting
    from matplotlib.axes import Axes
    result = bsk25_result
    before = {p.name: sha256(p) for p in result.data_path.iterdir()}
    labels = []
    original = Axes.plot
    def plot(self, *args, **kwargs):
        labels.append(kwargs.get("label"))
        return original(self, *args, **kwargs)
    with (
        patch.object(thermodynamics, "build_consistent_baseline", side_effect=AssertionError("solver")),
        patch.object(stellar, "solve_star", side_effect=AssertionError("solver")),
        patch.object(Axes, "plot", plot),
    ):
        path = plotting.plot_experiment(result.experiment_path, figures=["pressure"])
    assert "BSk25 (A=0)" in labels and "BSk24 (A=0)" not in labels
    assert (path / "figures.json").is_file()
    assert before == {p.name: sha256(p) for p in result.data_path.iterdir()}


def test_independent_reference_manifest():
    for line in (FIXTURE / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest


def test_selected_domain_amplitude_bounds_use_bsk25_endpoint():
    from eos_generation.assessment import calculate_windowed_amplitude_bounds
    eos = baseline.make_baseline_eos("bsk25")
    base = thermodynamics.build_consistent_baseline(
        thermodynamics.BSk24GridSettings(lower_points=257, upper_points=513), eos=eos,
    )
    bounds = calculate_windowed_amplitude_bounds(base, epsilon0_mev_fm3=200,
        sigma_mev_fm3=50, delta_mev_fm3=40, discovery_points=257)
    assert bounds.amplitude_min < 0 <= bounds.amplitude_max
    assert base.epsilon[-1] == eos.energy_density_max_causal_mev_fm3
    assert base.diagnostics["phase_and_composition_separation"]["compose_core_entry_n_fm3"] == REFERENCE["core_entry"]["n_b_fm3"]
    plan_experiment(ExperimentSettings.from_values(matter_model="bsk25", center=1900, width=30))
    with pytest.raises(ValueError, match="support"):
        plan_experiment(ExperimentSettings.from_values(center=1900, width=30))
