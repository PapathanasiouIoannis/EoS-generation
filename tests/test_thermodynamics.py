from __future__ import annotations

import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from eos_generation.bsk24 import baseline, deformation, reconstruction
from eos_generation._internal.planning import BSk24TrialConfig
from eos_generation._internal.thermodynamics import (
    _raw_gate_frame,
    _thermodynamic_profile_frame,
)
from eos_generation.bsk24._deformation_gate import _first_causal_crossing


ROOT = Path(__file__).resolve().parents[1]


class BSk24BaselineTests(unittest.TestCase):
    def test_analytic_fit_matches_retained_validated_regression_rows(self) -> None:
        rows = pd.read_csv(
            ROOT
            / "tests"
            / "fixtures"
            / "bsk24_contract_v1"
            / "thermodynamic_rows.csv"
        )
        direct = rows.loc[rows["case_id"] == "direct"]
        self.assertEqual(
            ["case_id", "epsilon_mev_fm3", "pressure_mev_fm3", "cs2"],
            rows.columns.tolist(),
        )
        self.assertEqual(3, len(direct))
        eos = baseline.make_bsk24_eos()
        epsilon = direct["epsilon_mev_fm3"].to_numpy(dtype=float)
        pressure = np.asarray(eos.pressure_from_energy_density(epsilon), dtype=float)
        density = epsilon * baseline.MEV_FM3_TO_MASS_DENSITY_G_CM3
        cs2 = np.asarray(eos.sound_speed_squared_from_mass_density(density), dtype=float)
        np.testing.assert_allclose(
            pressure,
            direct["pressure_mev_fm3"].to_numpy(dtype=float),
            rtol=5.0e-13,
            atol=0.0,
        )
        np.testing.assert_allclose(
            cs2,
            direct["cs2"].to_numpy(dtype=float),
            rtol=5.0e-13,
            atol=5.0e-14,
        )

    def test_published_fit_domain_is_monotone_stable_and_causal(self) -> None:
        eos = baseline.make_bsk24_eos()
        density = np.geomspace(
            baseline.FIT_MASS_DENSITY_MIN_G_CM3,
            baseline.CAUSAL_MASS_DENSITY_MAX_G_CM3,
            1001,
        )
        pressure = np.asarray(eos.pressure_from_mass_density(density))
        cs2 = np.asarray(eos.sound_speed_squared_from_mass_density(density))
        self.assertTrue(np.all(np.diff(pressure) > 0.0))
        self.assertTrue(np.all(cs2 > 0.0))
        self.assertTrue(np.all(cs2 <= 1.0))
        self.assertGreater(cs2[-1], 0.999)

    def test_forward_inverse_round_trip_and_no_extrapolation(self) -> None:
        eos = baseline.make_bsk24_eos()
        density = np.geomspace(1.0e7, 2.0e15, 101)
        pressure = np.asarray(eos.pressure_from_mass_density(density))
        recovered = np.asarray(eos.mass_density_from_pressure(pressure))
        self.assertLess(float(np.max(np.abs(recovered / density - 1.0))), 1.0e-11)
        with self.assertRaises(baseline.BSk24DomainError):
            eos.pressure_from_mass_density(
                np.nextafter(baseline.CAUSAL_MASS_DENSITY_MAX_G_CM3, math.inf)
            )


class WindowedDeformationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = reconstruction.build_consistent_baseline(
            reconstruction.BSk24GridSettings(lower_points=513, upper_points=1025)
        )
        cls.anchor = cls.base.anchor.energy_density_mev_fm3

    def test_window_has_exact_endpoint_derivatives(self) -> None:
        delta = 40.0
        for point, value in ((self.anchor, 0.0), (self.anchor + delta, 1.0)):
            self.assertEqual(
                value,
                deformation.smootherstep_window(
                    point,
                    epsilon_t_mev_fm3=self.anchor,
                    delta_mev_fm3=delta,
                ),
            )
            self.assertEqual(
                0.0,
                deformation.smootherstep_window_first_derivative(
                    point,
                    epsilon_t_mev_fm3=self.anchor,
                    delta_mev_fm3=delta,
                ),
            )
            self.assertEqual(
                0.0,
                deformation.smootherstep_window_second_derivative(
                    point,
                    epsilon_t_mev_fm3=self.anchor,
                    delta_mev_fm3=delta,
                ),
            )

    def test_tangential_continuous_causal_contact_is_an_endpoint(self) -> None:
        contact = _first_causal_crossing(
            np.asarray([1.0, 1.5, 2.5, 3.0]),
            lambda value: 1.0 - (float(value) - 2.0) ** 2,
            extrema_locations=(2.0,),
            xtol=1.0e-12,
            rtol=1.0e-12,
        )
        self.assertIsNotNone(contact)
        self.assertEqual(
            "resolved_first_continuous_causal_crossing",
            contact["status"],
        )
        self.assertEqual(2.0, contact["epsilon_mev_fm3"])
        self.assertEqual(1.0, contact["cs2_at_endpoint"])

        ambiguous = _first_causal_crossing(
            np.asarray([1.0, 1.5, 2.5, 3.0]),
            lambda value: 1.0 - 5.0e-14 - (float(value) - 2.0) ** 2,
            extrema_locations=(2.0,),
            xtol=1.0e-12,
            rtol=1.0e-12,
        )
        self.assertIsNotNone(ambiguous)
        self.assertEqual(
            "unresolved_near_tangential_causal_contact",
            ambiguous["status"],
        )
        self.assertFalse(
            ambiguous["crossing_included_to_governed_tolerance"]
        )

        superluminal = _first_causal_crossing(
            np.asarray([1.0, 1.5, 2.5, 3.0]),
            lambda value: 1.0 + 5.0e-15 - (float(value) - 2.0) ** 2,
            extrema_locations=(2.0,),
            xtol=1.0e-12,
            rtol=1.0e-12,
        )
        self.assertIsNotNone(superluminal)
        self.assertEqual(
            "resolved_first_continuous_causal_crossing",
            superluminal["status"],
        )
        self.assertLess(superluminal["epsilon_mev_fm3"], 2.0)
        self.assertLessEqual(superluminal["cs2_at_endpoint"], 1.0)
        self.assertTrue(
            superluminal["crossing_included_to_governed_tolerance"]
        )

    def test_a0_is_exact_array_identity(self) -> None:
        proposal = deformation.BSk24WindowedDeformation(
            "a0", 0.0, 200.0, 50.0, 40.0
        )
        gate, _, _ = deformation.raw_local_physics_gate(
            self.base,
            proposal,
            dense_lower_points=1025,
            dense_upper_points=4097,
        )
        eos = deformation.build_windowed_eos(self.base, proposal, raw_gate_report=gate)
        self.assertTrue(np.array_equal(eos.pressure, self.base.pressure))
        self.assertTrue(np.array_equal(eos.cs2, self.base.cs2))
        self.assertTrue(np.array_equal(eos.baryon_density, self.base.baryon_density))

    def test_a0_extreme_geometry_keeps_exact_identity_sampling(self) -> None:
        proposal = deformation.BSk24WindowedDeformation(
            "a0-extreme-geometry", 0.0, 200.0, 50.0, 1.0e-11
        )
        gate, _, _ = deformation.raw_local_physics_gate(
            self.base,
            proposal,
            dense_lower_points=17,
            dense_upper_points=17,
        )
        self.assertEqual("accepted_raw_local_physics_gate", gate["status"])
        self.assertEqual(
            "resolved_exact_zero_amplitude_identity_sampling",
            gate["continuous_resolution_certificate"]["status"],
        )
        self.assertEqual(
            0,
            gate["continuous_resolution_certificate"]["added_point_count"],
        )
        eos = deformation.build_windowed_eos(
            self.base,
            proposal,
            raw_gate_report=gate,
        )
        self.assertTrue(np.array_equal(eos.epsilon, self.base.epsilon))
        self.assertTrue(np.array_equal(eos.pressure, self.base.pressure))
        self.assertTrue(np.array_equal(eos.cs2, self.base.cs2))
        self.assertTrue(
            np.array_equal(eos.baryon_density, self.base.baryon_density)
        )

    def test_supplied_direct_gate_cannot_forge_an_arbitrary_truncation(self) -> None:
        proposal = deformation.BSk24WindowedDeformation(
            "forged-direct-endpoint", 0.0, 200.0, 50.0, 40.0
        )
        gate, _, _ = deformation.raw_local_physics_gate(
            self.base,
            proposal,
            dense_lower_points=257,
            dense_upper_points=513,
        )
        self.assertEqual("accepted_raw_local_physics_gate", gate["status"])
        self.assertEqual(
            "direct_bsk24_causal_endpoint",
            gate["retained_domain"]["endpoint_reason"],
        )
        forged = json.loads(json.dumps(gate))
        forged["retained_domain"]["epsilon_max_mev_fm3"] = 400.0
        with self.assertRaisesRegex(
            ValueError,
            "raw-gate retained endpoint",
        ):
            deformation.build_windowed_eos(
                self.base,
                proposal,
                raw_gate_report=forged,
            )

    def test_supplied_gate_cannot_forge_a_subcausal_crossing(self) -> None:
        proposal = deformation.BSk24WindowedDeformation(
            "forged-subcausal-crossing", 0.05, 200.0, 50.0, 40.0
        )
        gate, _, _ = deformation.raw_local_physics_gate(
            self.base,
            proposal,
            dense_lower_points=257,
            dense_upper_points=513,
        )
        forged = json.loads(json.dumps(gate))
        endpoint = 400.0
        endpoint_pressure = float(
            deformation._windowed_pressure(
                np.asarray([endpoint]), self.base, proposal
            )[0]
        )
        endpoint_cs2 = float(
            deformation._windowed_cs2(
                np.asarray([endpoint]), self.base, proposal
            )[0]
        )
        forged["full_retained_domain_passed"] = False
        forged[
            "complete_raw_proposal_causal_through_direct_endpoint"
        ] = False
        forged["retained_domain"].update(
            {
                "endpoint_reason": "first_continuous_causal_crossing",
                "epsilon_max_mev_fm3": endpoint,
                "pressure_max_mev_fm3": endpoint_pressure,
                "cs2_at_endpoint": endpoint_cs2,
                "first_causal_crossing": {
                    "status": "resolved_first_continuous_causal_crossing",
                    "epsilon_mev_fm3": endpoint,
                    "cs2_at_endpoint": endpoint_cs2,
                    "bracket_mev_fm3": [endpoint, endpoint],
                    "representable_bracket_width_mev_fm3": 0.0,
                    "governed_root_tolerance_mev_fm3": 1.0e-12,
                    "continuous_crossing_bracketed": True,
                    "crossing_included_to_governed_tolerance": True,
                    "cs2_values_modified": False,
                },
            }
        )
        self.assertLess(endpoint_cs2, 1.0)
        with self.assertRaisesRegex(
            ValueError,
            "raw-gate endpoint and first-crossing evidence disagree",
        ):
            deformation.build_windowed_eos(
                self.base,
                proposal,
                raw_gate_report=forged,
            )

    def test_near_tangential_gate_is_unresolved_but_superluminal_is_bracketed(
        self,
    ) -> None:
        threshold = 0.7701025843118475
        subcausal = deformation.BSk24WindowedDeformation(
            "near-subcausal",
            threshold - 5.0e-14,
            300.0,
            2.0,
            40.0,
        )
        subcausal_gate, _, _ = deformation.raw_local_physics_gate(
            self.base,
            subcausal,
            dense_lower_points=257,
            dense_upper_points=513,
        )
        self.assertEqual(
            "unresolved_raw_local_physics_gate", subcausal_gate["status"]
        )
        self.assertEqual(
            "unresolved_near_tangential_causal_contact",
            subcausal_gate["retained_domain"]["first_causal_crossing"][
                "status"
            ],
        )

        superluminal = deformation.BSk24WindowedDeformation(
            "slightly-superluminal",
            threshold + 5.0e-15,
            300.0,
            2.0,
            40.0,
        )
        superluminal_gate, _, _ = deformation.raw_local_physics_gate(
            self.base,
            superluminal,
            dense_lower_points=257,
            dense_upper_points=513,
        )
        crossing = superluminal_gate["retained_domain"][
            "first_causal_crossing"
        ]
        self.assertEqual(
            "accepted_raw_local_physics_gate", superluminal_gate["status"]
        )
        self.assertEqual(
            "brentq_estimate_plus_causal_side_float_refinement",
            crossing["refinement_method"],
        )
        self.assertLessEqual(crossing["cs2_at_endpoint"], 1.0)
        self.assertTrue(crossing["continuous_crossing_bracketed"])

    def test_pressure_primitive_matches_independent_quadrature(self) -> None:
        proposal = deformation.BSk24WindowedDeformation(
            "positive", 0.05, 200.0, 50.0, 40.0
        )
        for epsilon in (self.anchor + 0.1, self.anchor + 20.0, 200.0, 400.0):
            expected = proposal.amplitude * quad(
                lambda value: float(
                    deformation.windowed_gaussian_shape(
                        value,
                        proposal,
                        epsilon_t_mev_fm3=self.anchor,
                    )
                ),
                self.anchor,
                epsilon,
                epsabs=1.0e-13,
                epsrel=1.0e-13,
            )[0]
            observed = deformation.windowed_gaussian_pressure_primitive(
                epsilon,
                proposal,
                epsilon_t_mev_fm3=self.anchor,
            )
            self.assertAlmostEqual(expected, observed, delta=5.0e-12)

    def test_raw_invalid_candidate_is_rejected_without_repair(self) -> None:
        proposal = deformation.BSk24WindowedDeformation(
            "invalid", -1.0, 200.0, 50.0, 40.0
        )
        gate, _, _ = deformation.raw_local_physics_gate(
            self.base,
            proposal,
            dense_lower_points=1025,
            dense_upper_points=4097,
        )
        self.assertEqual("rejected_raw_local_physics_gate", gate["status"])
        self.assertEqual("none", gate["clipping_clamping_smoothing_repair"])
        with self.assertRaises(reconstruction.BSk24MechanicalStabilityError):
            deformation.build_windowed_eos(self.base, proposal, raw_gate_report=gate)

    def test_first_continuous_causal_crossing_is_accepted_and_consumed(self) -> None:
        proposal = deformation.BSk24WindowedDeformation(
            "early-causal", 1.0, 300.0, 2.0, 40.0
        )
        gate, raw_epsilon, raw_cs2 = deformation.raw_local_physics_gate(
            self.base,
            proposal,
            dense_lower_points=257,
            dense_upper_points=513,
        )

        self.assertEqual("accepted_raw_local_physics_gate", gate["status"])
        self.assertFalse(gate["full_retained_domain_passed"])
        retained = gate["retained_domain"]
        self.assertEqual(
            "first_continuous_causal_crossing",
            retained["endpoint_reason"],
        )
        self.assertTrue(
            retained["later_return_below_one_outside_usable_branch"]
        )
        endpoint = retained["epsilon_max_mev_fm3"]
        self.assertLess(endpoint, proposal.epsilon0_mev_fm3)
        self.assertEqual(
            self.base.eos.energy_density_max_published_fit_mev_fm3,
            raw_epsilon[-1],
        )
        self.assertTrue(np.any(raw_epsilon > endpoint))
        self.assertTrue(np.any(raw_cs2[raw_epsilon > endpoint] < 1.0))

        eos = deformation.build_windowed_eos(
            self.base, proposal, raw_gate_report=gate
        )
        self.assertEqual(endpoint, eos.epsilon[-1])
        self.assertEqual(endpoint, eos.energy_density_max_mev_fm3)
        self.assertEqual(
            "first_continuous_causal_crossing",
            eos.diagnostics["causal_domain"]["endpoint_reason"],
        )
        self.assertTrue(
            eos.diagnostics["causal_domain"]
            ["raw_gate_endpoint_consumed_without_rediscovery"]
        )
        self.assertLessEqual(eos.cs2[-1], 1.0)
        self.assertLess(abs(eos.cs2[-1] - 1.0), 1.0e-10)
        self.assertTrue(np.all(eos.cs2[:-1] < 1.0))
        crossing = retained["first_causal_crossing"]
        self.assertLessEqual(crossing["cs2_at_endpoint"], 1.0)
        self.assertTrue(crossing["crossing_included_to_governed_tolerance"])
        self.assertFalse(crossing["cs2_values_modified"])
        if crossing["first_noncausal_epsilon_mev_fm3"] is not None:
            self.assertGreater(crossing["first_noncausal_cs2"], 1.0)
            self.assertLessEqual(
                crossing["representable_bracket_width_mev_fm3"],
                crossing["governed_root_tolerance_mev_fm3"],
            )
        admissibility = deformation.full_domain_thermodynamic_admissibility(
            self.base,
            eos,
            raw_gate_report=gate,
        )
        self.assertEqual(
            "accepted_selected_domain_thermodynamic_gate",
            admissibility["status"],
        )
        self.assertEqual(
            "bsk24_selected_domain_thermodynamic_gate_v2",
            admissibility["schema_id"],
        )
        self.assertEqual(
            endpoint,
            admissibility["retained_domain_mev_fm3"][-1],
        )
        self.assertEqual(
            self.base.eos.energy_density_max_published_fit_mev_fm3,
            admissibility["complete_raw_domain_mev_fm3"][-1],
        )
        self.assertTrue(
            admissibility["independent_checks"]
            ["complete_raw_evidence_retained"]
        )
        self.assertTrue(
            admissibility["independent_checks"]
            ["selected_retained_domain_matches_raw_gate"]
        )
        self.assertEqual(
            "accepted_selected_domain_thermodynamic_gate",
            eos.diagnostics[
                "retained_domain_thermodynamic_admissibility"
            ]["status"],
        )

    def test_geometry_refinement_finds_narrow_between_node_island_and_pocket(
        self,
    ) -> None:
        left = self.base.anchor_index + 100
        center = float(
            0.5 * (self.base.epsilon[left] + self.base.epsilon[left + 1])
        )
        ordinary_spacing = float(
            self.base.epsilon[left + 1] - self.base.epsilon[left]
        )
        sigma = ordinary_spacing / 64.0
        self.assertLess(8.0 * sigma, ordinary_spacing)

        positive = deformation.BSk24WindowedDeformation(
            "narrow-island", 1.0, center, sigma, 40.0
        )
        ordinary_positive = self.base.cs2 + np.asarray(
            deformation.windowed_gaussian_delta_cs2(
                self.base.epsilon,
                positive,
                epsilon_t_mev_fm3=self.anchor,
            )
        )
        self.assertLessEqual(float(np.max(ordinary_positive)), 1.0)
        positive_gate, _, _ = deformation.raw_local_physics_gate(
            self.base,
            positive,
            dense_lower_points=129,
            dense_upper_points=257,
        )
        self.assertEqual(
            "accepted_raw_local_physics_gate", positive_gate["status"]
        )
        self.assertGreater(positive_gate["raw_maximum_cs2"], 1.0)
        positive_eos = deformation.build_windowed_eos(
            self.base, positive, raw_gate_report=positive_gate
        )
        analytical = positive_eos.diagnostics["tabulation_resolution"][
            "analytical_comparison"
        ]
        self.assertEqual("resolved_analytical_tabulation", analytical["status"])
        self.assertGreater(analytical["probe_count"], 0)

        negative = deformation.BSk24WindowedDeformation(
            "narrow-pocket", -1.0, center, sigma, 40.0
        )
        ordinary_negative = self.base.cs2 + np.asarray(
            deformation.windowed_gaussian_delta_cs2(
                self.base.epsilon,
                negative,
                epsilon_t_mev_fm3=self.anchor,
            )
        )
        self.assertGreater(float(np.min(ordinary_negative)), 0.0)
        negative_gate, _, _ = deformation.raw_local_physics_gate(
            self.base,
            negative,
            dense_lower_points=129,
            dense_upper_points=257,
        )
        self.assertEqual(
            "rejected_raw_local_physics_gate", negative_gate["status"]
        )
        self.assertLess(negative_gate["raw_minimum_cs2"], 0.0)
        self.assertEqual(
            "mechanical_stability_nonpositive_cs2",
            negative_gate["first_failure"]["reason"],
        )

    def test_four_sigma_overlap_is_total_below_and_above_domain(self) -> None:
        below_center = self.anchor - 20.0
        above_center = float(self.base.epsilon[-1] + 20.0)
        for center in (below_center, above_center):
            BSk24TrialConfig(
                amplitudes=(0.0, 0.01),
                epsilon0_mev_fm3=center,
                sigma_mev_fm3=10.0,
                deltas_mev_fm3=(40.0,),
            )
            proposal = deformation.BSk24WindowedDeformation(
                f"overlap-{center}", 0.01, center, 10.0, 40.0
            )
            gate, _, _ = deformation.raw_local_physics_gate(
                self.base,
                proposal,
                dense_lower_points=513,
                dense_upper_points=1025,
            )
            self.assertEqual("accepted_raw_local_physics_gate", gate["status"])

        self.assertEqual("evaluated", gate["raw_cs2_at_epsilon0_status"])
        with self.assertRaisesRegex(ValueError, "no meaningful in-domain support"):
            BSk24TrialConfig(
                amplitudes=(0.0, 0.01),
                epsilon0_mev_fm3=50.0,
                sigma_mev_fm3=10.0,
                deltas_mev_fm3=(40.0,),
            )
        no_support = deformation.BSk24WindowedDeformation(
            "no-support", 0.01, 50.0, 10.0, 40.0
        )
        no_support_gate, _, _ = deformation.raw_local_physics_gate(
            self.base,
            no_support,
            dense_lower_points=129,
            dense_upper_points=257,
        )
        self.assertEqual(
            "unresolved_raw_local_physics_gate", no_support_gate["status"]
        )
        json.dumps(no_support_gate, allow_nan=False)
        with self.assertRaisesRegex(ValueError, "no meaningful in-domain support"):
            BSk24TrialConfig(
                amplitudes=(0.0, 0.01),
                epsilon0_mev_fm3=1600.0,
                sigma_mev_fm3=10.0,
                deltas_mev_fm3=(40.0,),
            )

    def test_amplitude_bounds_resolve_a_narrow_below_anchor_tail(self) -> None:
        sigma = 1.0e-3
        center = self.anchor - 2.0 * sigma
        delta = 40.0
        bounds = deformation.calculate_windowed_amplitude_bounds(
            self.base,
            epsilon0_mev_fm3=center,
            sigma_mev_fm3=sigma,
            delta_mev_fm3=delta,
            discovery_points=257,
        )

        support_upper = center + 4.0 * sigma
        scale = support_upper - self.anchor
        unit_proposal = deformation.BSk24WindowedDeformation(
            "independent-bound-probe",
            1.0,
            center,
            sigma,
            delta,
        )

        def independent_log_ratio(normalized: float) -> float:
            epsilon = self.anchor + float(normalized) * scale
            if epsilon <= self.anchor:
                epsilon = math.nextafter(self.anchor, support_upper)
            density = epsilon * baseline.MEV_FM3_TO_MASS_DENSITY_G_CM3
            direct_cs2 = float(
                self.base.eos.sound_speed_squared_from_mass_density(density)
            )
            shape = float(
                deformation.windowed_gaussian_shape(
                    epsilon,
                    unit_proposal,
                    epsilon_t_mev_fm3=self.anchor,
                )
            )
            return math.log(direct_cs2) - math.log(shape)

        independent = minimize_scalar(
            independent_log_ratio,
            bounds=(0.0, 1.0),
            method="bounded",
            options={"xatol": 1.0e-13},
        )
        self.assertTrue(independent.success)
        expected_minimum = -math.exp(float(independent.fun))
        self.assertLess(
            abs(bounds.amplitude_min / expected_minimum - 1.0),
            1.0e-8,
        )
        self.assertLess(abs(bounds.amplitude_min), 1.0e15)
        self.assertGreater(
            bounds.lower_limiting_epsilon_mev_fm3,
            self.anchor,
        )
        self.assertLess(
            bounds.lower_limiting_epsilon_mev_fm3,
            support_upper,
        )

    def test_raw_gate_table_preserves_analytical_pressure_proposal(self) -> None:
        proposal = deformation.BSk24WindowedDeformation(
            "raw-pressure", 0.05, 200.0, 50.0, 40.0
        )
        gate, epsilon, raw_cs2 = deformation.raw_local_physics_gate(
            self.base,
            proposal,
            dense_lower_points=129,
            dense_upper_points=257,
        )
        frame = _raw_gate_frame(
            case_id=proposal.case_id,
            deformation=proposal,
            baseline=self.base,
            epsilon=epsilon,
            raw_cs2=raw_cs2,
            status=gate["status"],
        )
        for column in (
            "direct_pressure_mev_fm3",
            "delta_pressure_mev_fm3",
            "raw_pressure_mev_fm3",
        ):
            self.assertIn(column, frame)
        np.testing.assert_array_equal(
            frame["raw_pressure_mev_fm3"].to_numpy(),
            frame["direct_pressure_mev_fm3"].to_numpy()
            + frame["delta_pressure_mev_fm3"].to_numpy(),
        )

    def test_nonmonotone_analytical_pressure_is_unresolved_at_raw_gate(
        self,
    ) -> None:
        proposal = deformation.BSk24WindowedDeformation(
            "far-tail-cancellation",
            0.32304095453401915,
            726.7389802403162,
            69.87981314665237,
            1.0718924105447882,
        )
        gate, epsilon, raw_cs2 = deformation.raw_local_physics_gate(
            self.base,
            proposal,
            dense_lower_points=17,
            dense_upper_points=17,
        )
        certificate = gate["raw_pressure_reconstruction_certificate"]
        self.assertEqual("unresolved_raw_local_physics_gate", gate["status"])
        self.assertEqual(
            "unresolved_raw_pressure_cs2_consistency",
            gate["first_failure"]["reason"],
        )
        self.assertEqual(
            "unresolved_raw_pressure_cs2_consistency",
            certificate["status"],
        )
        self.assertLess(
            certificate["minimum_forward_pressure_difference_mev_fm3"],
            0.0,
        )
        self.assertFalse(certificate["pressure_values_modified"])
        self.assertFalse(gate["strictly_monotone_pressure_implied"])
        frame = _raw_gate_frame(
            case_id=proposal.case_id,
            deformation=proposal,
            baseline=self.base,
            epsilon=epsilon,
            raw_cs2=raw_cs2,
            status=gate["status"],
        )
        self.assertLess(
            float(np.min(np.diff(frame["raw_pressure_mev_fm3"]))),
            0.0,
        )
        with self.assertRaises(
            reconstruction.BSk24MechanicalStabilityError
        ) as caught:
            deformation.build_windowed_eos(
                self.base,
                proposal,
                raw_gate_report=gate,
            )
        self.assertEqual(
            "unresolved_raw_local_physics_gate",
            caught.exception.diagnostics["status"],
        )

    def test_unresolved_raw_pressure_derivative_never_reaches_build(
        self,
    ) -> None:
        proposal = deformation.BSk24WindowedDeformation(
            "unresolved-ramp-tabulation",
            -0.48849463387908165,
            1292.1852372022038,
            137.64908205163843,
            0.1349739042127701,
        )
        gate, _, _ = deformation.raw_local_physics_gate(
            self.base,
            proposal,
            dense_lower_points=17,
            dense_upper_points=17,
        )
        certificate = gate["raw_pressure_reconstruction_certificate"]
        comparison = certificate["analytical_derivative_comparison"]
        self.assertEqual("unresolved_raw_local_physics_gate", gate["status"])
        self.assertEqual(
            "unresolved_raw_pressure_cs2_consistency",
            gate["first_failure"]["reason"],
        )
        self.assertEqual(
            "unresolved_analytical_tabulation",
            comparison["status"],
        )
        self.assertGreater(
            comparison["maximum_absolute_error"],
            comparison["maximum_allowed_absolute_error"],
        )
        self.assertFalse(
            comparison["pressure_or_cs2_values_modified"]
        )
        with self.assertRaises(
            reconstruction.BSk24MechanicalStabilityError
        ) as caught:
            deformation.build_windowed_eos(
                self.base,
                proposal,
                raw_gate_report=gate,
            )
        self.assertEqual(
            "unresolved_raw_local_physics_gate",
            caught.exception.diagnostics["status"],
        )

    def test_retained_tabulation_is_certified_before_raw_acceptance(
        self,
    ) -> None:
        proposal = deformation.BSk24WindowedDeformation(
            "retained-resolution", 100.0, 300.0, 2.0, 40.0
        )
        gate, _, _ = deformation.raw_local_physics_gate(
            self.base,
            proposal,
            dense_lower_points=257,
            dense_upper_points=513,
        )
        certificate = gate["retained_tabulation_resolution_certificate"]
        self.assertEqual("unresolved_raw_local_physics_gate", gate["status"])
        self.assertEqual(
            "unresolved_retained_tabulation_resolution",
            gate["first_failure"]["reason"],
        )
        self.assertEqual(
            "unresolved_tabulation_resolution", certificate["status"]
        )
        self.assertFalse(gate["retained_domain"]["resolution_certified"])
        self.assertFalse(certificate["reconstruction_performed"])
        self.assertFalse(certificate["stellar_work_performed"])
        with self.assertRaises(
            reconstruction.BSk24MechanicalStabilityError
        ) as caught:
            deformation.build_windowed_eos(
                self.base,
                proposal,
                raw_gate_report=gate,
            )
        self.assertEqual(
            "unresolved_raw_local_physics_gate",
            caught.exception.diagnostics["status"],
        )

    def test_unresolved_tabulation_and_unusable_inverse_fail_closed(self) -> None:
        proposal = deformation.BSk24WindowedDeformation(
            "resolution", 0.05, 200.0, 50.0, 40.0
        )
        gate, _, _ = deformation.raw_local_physics_gate(
            self.base,
            proposal,
            dense_lower_points=129,
            dense_upper_points=257,
        )
        unresolved = {
            "status": "unresolved_tabulation_resolution",
            "failure_reason": "synthetic_bounded_resolution_exhaustion",
        }
        with (
            patch.object(
                deformation,
                "_retained_resolution_grid",
                return_value=(self.base.epsilon.copy(), unresolved),
            ),
            self.assertRaises(reconstruction.BSk24MechanicalStabilityError) as caught,
        ):
            deformation.build_windowed_eos(
                self.base, proposal, raw_gate_report=gate
            )
        self.assertEqual(
            "unresolved_tabulation_resolution",
            caught.exception.diagnostics["status"],
        )
        self.assertFalse(caught.exception.diagnostics["stellar_work_permitted"])

        with (
            patch.object(
                deformation.BSk24WindowedEos,
                "energy_density_from_pressure",
                return_value=np.asarray([math.nan]),
            ),
            patch.object(
                deformation.BSk24WindowedEos,
                "pressure_from_energy_density",
                return_value=np.asarray([math.nan]),
            ),
            self.assertRaises(reconstruction.BSk24MechanicalStabilityError) as caught,
        ):
            deformation.build_windowed_eos(
                self.base, proposal, raw_gate_report=gate
            )
        self.assertEqual(
            "rejected_unusable_reconstruction_inversion",
            caught.exception.diagnostics["status"],
        )

    def test_finite_diagnostic_magnitudes_do_not_reject_reconstruction(self) -> None:
        proposal = deformation.BSk24WindowedDeformation(
            "finite-diagnostics", 0.05, 200.0, 50.0, 40.0
        )
        gate, _, _ = deformation.raw_local_physics_gate(
            self.base,
            proposal,
            dense_lower_points=129,
            dense_upper_points=257,
        )
        original = deformation._residual_arrays

        def large_finite_residuals(*args: object, **kwargs: object):
            result = original(*args, **kwargs)
            result["r_c"] = np.full_like(result["r_c"], 1.0e6)
            return result

        with patch.object(
            deformation,
            "_residual_arrays",
            side_effect=large_finite_residuals,
        ):
            eos = deformation.build_windowed_eos(
                self.base, proposal, raw_gate_report=gate
            )
        self.assertEqual(1.0e6, float(np.max(eos.residuals["r_c"])))
        self.assertEqual(
            "resolved_finite_monotone_nonextrapolating",
            eos.diagnostics["tabulation_resolution"]
            ["interpolation_inversion_status"],
        )

    def test_signed_response_is_preserved_below_and_changes_above_anchor(self) -> None:
        results = []
        for amplitude in (-0.05, 0.05):
            proposal = deformation.BSk24WindowedDeformation(
                str(amplitude), amplitude, 200.0, 50.0, 40.0
            )
            gate, _, _ = deformation.raw_local_physics_gate(
                self.base,
                proposal,
                dense_lower_points=1025,
                dense_upper_points=4097,
            )
            results.append(
                deformation.build_windowed_eos(self.base, proposal, raw_gate_report=gate)
            )
        for eos in results:
            self.assertTrue(
                np.array_equal(
                    eos.pressure[: self.base.anchor_index],
                    self.base.pressure[: self.base.anchor_index],
                )
            )
        for eos, expected_sign in zip(results, (-1.0, 1.0)):
            mask = (eos.epsilon > self.anchor) & (
                eos.epsilon <= self.base.epsilon[-1]
            )
            direct = np.asarray(
                self.base.eos.pressure_from_energy_density(
                    eos.epsilon[mask]
                ),
                dtype=float,
            )
            response = eos.pressure[mask] - direct
            if expected_sign < 0.0:
                self.assertLess(np.min(response), 0.0)
            else:
                self.assertGreater(np.max(response), 0.0)

    def test_negative_deformation_extends_to_its_combined_causal_endpoint(self) -> None:
        proposal = deformation.BSk24WindowedDeformation(
            "extended-negative", -0.08, 200.0, 150.0, 250.0
        )
        gate, raw_epsilon, _ = deformation.raw_local_physics_gate(
            self.base,
            proposal,
            dense_lower_points=1025,
            dense_upper_points=4097,
        )
        self.assertEqual("accepted_raw_local_physics_gate", gate["status"])
        self.assertEqual(
            self.base.eos.energy_density_max_published_fit_mev_fm3,
            raw_epsilon[-1],
        )
        retained = gate["retained_domain"]
        self.assertEqual(
            "first_continuous_causal_crossing",
            retained["endpoint_reason"],
        )
        self.assertGreater(
            retained["epsilon_max_mev_fm3"], self.base.epsilon[-1]
        )
        eos = deformation.build_windowed_eos(
            self.base, proposal, raw_gate_report=gate
        )
        self.assertEqual(retained["epsilon_max_mev_fm3"], eos.epsilon[-1])
        self.assertLessEqual(eos.cs2[-1], 1.0)
        frame = _thermodynamic_profile_frame(
            self.base, {proposal.case_id: eos}
        )
        extended = frame.loc[frame["case_id"] == proposal.case_id]
        self.assertTrue(
            np.isfinite(
                extended[
                    [
                        "pressure_relative_to_direct",
                        "baryon_density_relative_to_direct",
                        "enthalpy_relative_to_direct",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
