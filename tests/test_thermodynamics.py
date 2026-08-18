from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from scipy.integrate import quad

from eos_generation.bsk24 import baseline, deformation, reconstruction


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
        mask = self.base.epsilon > self.anchor
        self.assertLess(np.min(results[0].pressure[mask] - self.base.pressure[mask]), 0.0)
        self.assertGreater(np.max(results[1].pressure[mask] - self.base.pressure[mask]), 0.0)


if __name__ == "__main__":
    unittest.main()
