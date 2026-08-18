"""Approved unified BSk24 analytical baseline from Pearson et al. (2018).

The production barotrope is Appendix C, equation (C4), with the BSk24
coefficients in table C2 and ``K=-33.2047`` so pressure is returned in
MeV/fm^3.  The 2019 erratum governs the source record but does not change
Appendix C.  No table interpolation, crust splice, smoothing, or causal
repair is performed here.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import brentq


MODEL_NAME = "BSK24_ANALYTIC_PEARSON2018_CORR2019"
MODEL_VERSION = "pearson2018-corr2019-v1"
VALIDATION_STATUS = "pass"
VALIDATION_SCOPE = {
    "status_applies_to": (
        "declared_Pearson_Equation_C4_pressure_and_analytical_derivative_representation",
        "declared_Pearson_Equation_C1_anchor_use",
        "effective_first_law_thermodynamic_reconstruction",
        "stated_units_and_published_fit_plus_retained_causal_domain",
    ),
    "status_does_not_imply_validation_of": (
        "perturbed_microscopic_composition",
        "species_chemical_potentials",
        "exact_zero_pressure_surface_behavior",
        "resolved_maximum_mass",
        "radial_modes",
        "observational_compatibility",
        "equality_of_the_analytical_fit_and_CompOSE_table",
    ),
}

C_LIGHT_CM_S = 2.99792458e10
MEV_FM3_TO_ERG_CM3 = 1.602176634e33
MEV_FM3_TO_MASS_DENSITY_G_CM3 = MEV_FM3_TO_ERG_CM3 / C_LIGHT_CM_S**2

FIT_LOG10_MASS_DENSITY_MIN = 6.0
FIT_LOG10_MASS_DENSITY_MAX = 16.0
FIT_MASS_DENSITY_MIN_G_CM3 = 10.0**FIT_LOG10_MASS_DENSITY_MIN
FIT_MASS_DENSITY_MAX_G_CM3 = 10.0**FIT_LOG10_MASS_DENSITY_MAX
CAUSAL_MASS_DENSITY_MAX_G_CM3 = 2.69e15
CAUSAL_BARYON_DENSITY_MAX_FM3 = 1.088

PRESSURE_LOG10_OFFSET_MEV_FM3 = -33.2047
NEUTRON_REST_ENERGY_MEV = 939.5654
IRON56_GROUND_OFFSET_MEV = -9.1536

# Pearson et al. (2018), table C2, BSk24 row; also present verbatim in the
# official Ioffe bskfit18.f revision 2023-02-13 for KEOS=24.
PRESSURE_COEFFICIENTS = (
    6.795,
    5.552,
    0.00435,
    0.13963,
    3.636,
    11.943,
    13.848,
    1.3031,
    3.644,
    -30.840,
    2.2322,
    4.65,
    14.290,
    30.08,
    -2.080,
    1.10,
    14.71,
    0.099,
    11.66,
    5.00,
    -0.095,
    14.15,
    9.1,
)

# Pearson et al. (2018), table C1, BSk24 row; used only for the approved
# auxiliary baryon-density/energy-per-baryon state.
ENERGY_COEFFICIENTS = (
    6.590e8,
    9.49e10,
    6.95e7,
    5.63e6,
    6.51e5,
    19.37,
    0.1028,
    4.09,
    6726.0,
    29.57,
    4.39,
    19.51,
    2.6728,
    1.75,
)


class BSk24DomainError(ValueError):
    """Raised when an evaluation would leave the approved retained domain."""


class BSk24InversionError(RuntimeError):
    """Raised when a bracketed BSk24 inverse fails to converge."""


def _scalar_or_array(value: np.ndarray) -> float | np.ndarray:
    return float(value) if value.ndim == 0 else value


def _fermi(argument: np.ndarray) -> np.ndarray:
    """Stable evaluation of ``1 / (exp(argument) + 1)``."""
    return np.exp(-np.logaddexp(0.0, argument))


def _log10_pressure_from_xi(xi: np.ndarray, *, offset: float) -> np.ndarray:
    a = PRESSURE_COEFFICIENTS
    denominator = 1.0 + a[3] * xi
    rational = (a[0] + a[1] * xi + a[2] * xi**3) / denominator
    result = rational * _fermi(a[4] * (xi - a[5]))
    result += (a[6] + a[7] * xi) * _fermi(a[8] * (a[5] - xi))
    result += (a[9] + a[10] * xi) * _fermi(a[11] * (a[12] - xi))
    result += (a[13] + a[14] * xi) * _fermi(a[15] * (a[16] - xi))
    result += a[17] / (1.0 + (a[19] * (xi - a[18])) ** 2)
    result += a[20] / (1.0 + (a[22] * (xi - a[21])) ** 2)
    return result + offset


def _dlog10_pressure_dxi(xi: np.ndarray) -> np.ndarray:
    """Analytical derivative of Appendix-C equation (C4)."""
    a = PRESSURE_COEFFICIENTS
    denominator = 1.0 + a[3] * xi
    numerator = a[0] + a[1] * xi + a[2] * xi**3
    numerator_prime = a[1] + 3.0 * a[2] * xi**2
    rational = numerator / denominator
    rational_prime = (
        numerator_prime * denominator - numerator * a[3]
    ) / denominator**2

    f1 = _fermi(a[4] * (xi - a[5]))
    f2 = _fermi(a[8] * (a[5] - xi))
    f3 = _fermi(a[11] * (a[12] - xi))
    f4 = _fermi(a[15] * (a[16] - xi))
    result = rational_prime * f1 - rational * a[4] * f1 * (1.0 - f1)
    result += a[7] * f2 + (a[6] + a[7] * xi) * a[8] * f2 * (1.0 - f2)
    result += a[10] * f3 + (a[9] + a[10] * xi) * a[11] * f3 * (1.0 - f3)
    result += a[14] * f4 + (a[13] + a[14] * xi) * a[15] * f4 * (1.0 - f4)

    delta5 = xi - a[18]
    delta6 = xi - a[21]
    result -= 2.0 * a[17] * a[19] ** 2 * delta5 / (1.0 + (a[19] * delta5) ** 2) ** 2
    result -= 2.0 * a[20] * a[22] ** 2 * delta6 / (1.0 + (a[22] * delta6) ** 2) ** 2
    return result


def _energy_per_baryon_above_iron_ground_mev(
    baryon_density_fm3: np.ndarray,
) -> np.ndarray:
    """Pearson et al. Appendix-C equation (C1), BSk24 coefficients."""
    a = ENERGY_COEFFICIENTS
    n = baryon_density_fm3
    high = (a[9] * n) ** a[12] / (1.0 + a[11] * n)
    high_weight = 1.0 / (1.0 + (a[10] * n) ** a[13])
    middle = a[5] * n ** a[6] * (1.0 + a[7] * n)
    middle_weight = 1.0 / (1.0 + a[8] * n)
    low = (a[0] * n) ** 1.16667 / (1.0 + np.sqrt(a[1] * n))
    low *= (1.0 + np.sqrt(a[3] * n)) / (1.0 + np.sqrt(a[2] * n))
    low /= 1.0 + np.sqrt(a[4] * n)
    return low * middle_weight + middle * (1.0 - middle_weight) * high_weight + high * (
        1.0 - high_weight
    )


class BSk24AnalyticEos:
    """Causal-domain production adapter for the approved unified BSk24 fit."""

    model_name = MODEL_NAME
    model_version = MODEL_VERSION
    validation_status = VALIDATION_STATUS
    validation_scope = VALIDATION_SCOPE
    eps_surf = 0.0
    discontinuities: tuple[()] = ()
    requires_discontinuity_metadata = False

    @staticmethod
    def _require_range(
        value: Any,
        *,
        name: str,
        lower: float,
        upper: float,
    ) -> np.ndarray:
        try:
            values = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise BSk24DomainError(f"{name} must be numeric") from exc
        if not np.all(np.isfinite(values)):
            raise BSk24DomainError(f"{name} must contain only finite values")
        if np.any(values < lower) or np.any(values > upper):
            raise BSk24DomainError(
                f"{name} is outside the approved retained interval [{lower!r}, {upper!r}]"
            )
        return values

    @property
    def pressure_min_mev_fm3(self) -> float:
        return float(
            10.0
            ** _log10_pressure_from_xi(
                np.asarray(FIT_LOG10_MASS_DENSITY_MIN),
                offset=PRESSURE_LOG10_OFFSET_MEV_FM3,
            )
        )

    @property
    def pressure_max_causal_mev_fm3(self) -> float:
        return float(
            10.0
            ** _log10_pressure_from_xi(
                np.asarray(math.log10(CAUSAL_MASS_DENSITY_MAX_G_CM3)),
                offset=PRESSURE_LOG10_OFFSET_MEV_FM3,
            )
        )

    @property
    def energy_density_min_mev_fm3(self) -> float:
        return FIT_MASS_DENSITY_MIN_G_CM3 / MEV_FM3_TO_MASS_DENSITY_G_CM3

    @property
    def energy_density_max_causal_mev_fm3(self) -> float:
        return CAUSAL_MASS_DENSITY_MAX_G_CM3 / MEV_FM3_TO_MASS_DENSITY_G_CM3

    def pressure_from_mass_density(self, mass_density_g_cm3: Any) -> float | np.ndarray:
        rho = self._require_range(
            mass_density_g_cm3,
            name="mass_density_g_cm3",
            lower=FIT_MASS_DENSITY_MIN_G_CM3,
            upper=CAUSAL_MASS_DENSITY_MAX_G_CM3,
        )
        pressure = 10.0 ** _log10_pressure_from_xi(
            np.log10(rho),
            offset=PRESSURE_LOG10_OFFSET_MEV_FM3,
        )
        return _scalar_or_array(pressure)

    def diagnostic_pressure_from_mass_density(
        self,
        mass_density_g_cm3: Any,
    ) -> float | np.ndarray:
        """Evaluate the published fit above causality only for explicit diagnostics."""
        rho = self._require_range(
            mass_density_g_cm3,
            name="diagnostic_mass_density_g_cm3",
            lower=FIT_MASS_DENSITY_MIN_G_CM3,
            upper=FIT_MASS_DENSITY_MAX_G_CM3,
        )
        pressure = 10.0 ** _log10_pressure_from_xi(
            np.log10(rho),
            offset=PRESSURE_LOG10_OFFSET_MEV_FM3,
        )
        return _scalar_or_array(pressure)

    def diagnostic_pressure_dyn_cm2_from_mass_density(
        self,
        mass_density_g_cm3: Any,
    ) -> float | np.ndarray:
        """Return the same published fit with ``K=0`` in dyn/cm^2."""
        rho = self._require_range(
            mass_density_g_cm3,
            name="diagnostic_mass_density_g_cm3",
            lower=FIT_MASS_DENSITY_MIN_G_CM3,
            upper=FIT_MASS_DENSITY_MAX_G_CM3,
        )
        pressure = 10.0 ** _log10_pressure_from_xi(np.log10(rho), offset=0.0)
        return _scalar_or_array(pressure)

    def pressure_from_energy_density(self, energy_density_mev_fm3: Any) -> float | np.ndarray:
        epsilon = self._require_range(
            energy_density_mev_fm3,
            name="total_energy_density_mev_fm3",
            lower=self.energy_density_min_mev_fm3,
            upper=self.energy_density_max_causal_mev_fm3,
        )
        rho = epsilon * MEV_FM3_TO_MASS_DENSITY_G_CM3
        pressure = 10.0 ** _log10_pressure_from_xi(
            np.log10(rho),
            offset=PRESSURE_LOG10_OFFSET_MEV_FM3,
        )
        return _scalar_or_array(pressure)

    def mass_density_from_pressure(self, pressure_mev_fm3: Any) -> float | np.ndarray:
        pressure = self._require_range(
            pressure_mev_fm3,
            name="pressure_mev_fm3",
            lower=self.pressure_min_mev_fm3,
            upper=self.pressure_max_causal_mev_fm3,
        )
        flat = pressure.reshape(-1)
        densities = np.empty_like(flat)
        lower = FIT_LOG10_MASS_DENSITY_MIN
        upper = math.log10(CAUSAL_MASS_DENSITY_MAX_G_CM3)
        lower_pressure = self.pressure_min_mev_fm3
        upper_pressure = self.pressure_max_causal_mev_fm3
        for index, target in enumerate(flat):
            if target == lower_pressure:
                densities[index] = FIT_MASS_DENSITY_MIN_G_CM3
                continue
            if target == upper_pressure:
                densities[index] = CAUSAL_MASS_DENSITY_MAX_G_CM3
                continue
            log_target = math.log10(float(target))
            try:
                root = brentq(
                    lambda xi: float(
                        _log10_pressure_from_xi(np.asarray(xi), offset=PRESSURE_LOG10_OFFSET_MEV_FM3)
                    )
                    - log_target,
                    lower,
                    upper,
                    xtol=5.0e-14,
                    rtol=4.0 * np.finfo(float).eps,
                )
                densities[index] = 10.0**root
            except (ValueError, RuntimeError) as exc:
                raise BSk24InversionError(
                    f"BSk24 pressure inversion failed for {target!r} MeV/fm^3"
                ) from exc
        result = densities.reshape(pressure.shape)
        return _scalar_or_array(result)

    def energy_density_from_pressure(self, pressure_mev_fm3: Any) -> float | np.ndarray:
        rho = np.asarray(self.mass_density_from_pressure(pressure_mev_fm3), dtype=float)
        epsilon = rho / MEV_FM3_TO_MASS_DENSITY_G_CM3
        return _scalar_or_array(epsilon)

    def log_derivative_pressure_mass_density(
        self,
        mass_density_g_cm3: Any,
    ) -> float | np.ndarray:
        rho = self._require_range(
            mass_density_g_cm3,
            name="mass_density_g_cm3",
            lower=FIT_MASS_DENSITY_MIN_G_CM3,
            upper=CAUSAL_MASS_DENSITY_MAX_G_CM3,
        )
        derivative = _dlog10_pressure_dxi(np.log10(rho))
        return _scalar_or_array(derivative)

    def sound_speed_squared_from_mass_density(
        self,
        mass_density_g_cm3: Any,
    ) -> float | np.ndarray:
        rho = self._require_range(
            mass_density_g_cm3,
            name="mass_density_g_cm3",
            lower=FIT_MASS_DENSITY_MIN_G_CM3,
            upper=CAUSAL_MASS_DENSITY_MAX_G_CM3,
        )
        pressure = np.asarray(self.pressure_from_mass_density(rho), dtype=float)
        epsilon = rho / MEV_FM3_TO_MASS_DENSITY_G_CM3
        result = pressure * _dlog10_pressure_dxi(np.log10(rho)) / epsilon
        return _scalar_or_array(result)

    def sound_speed_squared_from_pressure(self, pressure_mev_fm3: Any) -> float | np.ndarray:
        rho = self.mass_density_from_pressure(pressure_mev_fm3)
        return self.sound_speed_squared_from_mass_density(rho)

    def _mass_density_from_baryon_density(self, baryon_density_fm3: float) -> float:
        n = np.asarray(float(baryon_density_fm3))
        energy = _energy_per_baryon_above_iron_ground_mev(n)
        total_per_baryon = NEUTRON_REST_ENERGY_MEV + IRON56_GROUND_OFFSET_MEV + energy
        return float(n * total_per_baryon * MEV_FM3_TO_MASS_DENSITY_G_CM3)

    def baryon_density_from_mass_density(self, mass_density_g_cm3: Any) -> float | np.ndarray:
        rho = self._require_range(
            mass_density_g_cm3,
            name="mass_density_g_cm3",
            lower=FIT_MASS_DENSITY_MIN_G_CM3,
            upper=CAUSAL_MASS_DENSITY_MAX_G_CM3,
        )
        flat = rho.reshape(-1)
        result = np.empty_like(flat)
        for index, target in enumerate(flat):
            try:
                result[index] = brentq(
                    lambda n: self._mass_density_from_baryon_density(n) - float(target),
                    1.0e-12,
                    2.0,
                    xtol=5.0e-15,
                    rtol=4.0 * np.finfo(float).eps,
                )
            except (ValueError, RuntimeError) as exc:
                raise BSk24InversionError(
                    f"BSk24 baryon-density inversion failed for {target!r} g/cm^3"
                ) from exc
        return _scalar_or_array(result.reshape(rho.shape))

    def source_state_from_mass_density(self, mass_density_g_cm3: Any) -> dict[str, Any]:
        rho = self._require_range(
            mass_density_g_cm3,
            name="mass_density_g_cm3",
            lower=FIT_MASS_DENSITY_MIN_G_CM3,
            upper=CAUSAL_MASS_DENSITY_MAX_G_CM3,
        )
        n = np.asarray(self.baryon_density_from_mass_density(rho), dtype=float)
        energy_above_ground = _energy_per_baryon_above_iron_ground_mev(n)
        energy_above_neutron = energy_above_ground + IRON56_GROUND_OFFSET_MEV
        return {
            "mass_density_g_cm3": _scalar_or_array(rho),
            "total_energy_density_mev_fm3": _scalar_or_array(
                rho / MEV_FM3_TO_MASS_DENSITY_G_CM3
            ),
            "pressure_mev_fm3": self.pressure_from_mass_density(rho),
            "baryon_density_fm3": _scalar_or_array(n),
            "energy_per_baryon_above_iron56_ground_mev": _scalar_or_array(
                energy_above_ground
            ),
            "total_energy_per_baryon_minus_neutron_rest_mev": _scalar_or_array(
                energy_above_neutron
            ),
            "baryon_chemical_potential": None,
        }

    def __call__(self, pressure_mev_fm3: float) -> tuple[float, float]:
        pressure = float(pressure_mev_fm3)
        # A TOV right-hand-side evaluation needs both epsilon(P) and
        # c_s^2(P).  Both public convenience methods invert the same pressure,
        # so composing them here used to run the governed Brent inversion
        # twice.  Reuse the one exact root and evaluate both quantities from
        # that mass density; the public methods and their behavior remain
        # unchanged.
        mass_density = float(self.mass_density_from_pressure(pressure))
        epsilon = mass_density / MEV_FM3_TO_MASS_DENSITY_G_CM3
        cs2 = float(
            self.sound_speed_squared_from_mass_density(mass_density)
        )
        return epsilon, cs2

    def provenance(self) -> dict[str, Any]:
        """Return complete JSON-safe model identity and scientific conventions."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "source_publication": {
                "primary": "Pearson et al., MNRAS 481, 2994-3026 (2018), Appendix C",
                "erratum": "Pearson et al., MNRAS 486, 768 (2019)",
            },
            "source_artifact_or_routine": {
                "independent_implementation_oracle": "Ioffe bskfit18.f revision 2023-02-13, KEOS=24",
                "underlying_tabulated_oracle": "CompOSE PCP(BSK24), EoS ID 253",
            },
            "source_checksums_sha256": {
                "ioffe_bskfit18_f": "4bd3b716f04e40c69165fa83b5cd8ecb03c30aa1f2b45342fb070cef09a3a99c",
                "compose_eos_zip": "5db3e010372805f065f04676982c4127203a4bb7b4dc15d25090e9f34bed6582",
            },
            "retrieval_date": "2026-07-22",
            "representation_type": "published_analytical_fit_unified_crust_and_core",
            "units": {
                "pressure": "MeV/fm^3",
                "mass_density": "g/cm^3",
                "total_energy_density": "MeV/fm^3",
                "baryon_density": "fm^-3",
                "energy_per_baryon": "MeV",
                "sound_speed_squared": "dimensionless_c_equals_1",
            },
            "thermodynamic_conventions": {
                "mass_density": "total_mass_energy_density_including_rest_mass",
                "total_energy_density": "epsilon=rho*c^2_including_rest_mass",
                "internal_energy_density": "not_exposed_as_an_independent_barotrope_variable",
                "baryon_chemical_potential": "unavailable_not_reconstructed",
            },
            "valid_domain": {
                "published_fit_mass_density_g_cm3": [
                    FIT_MASS_DENSITY_MIN_G_CM3,
                    FIT_MASS_DENSITY_MAX_G_CM3,
                ],
                "production_retained_mass_density_g_cm3": [
                    FIT_MASS_DENSITY_MIN_G_CM3,
                    CAUSAL_MASS_DENSITY_MAX_G_CM3,
                ],
                "production_retained_pressure_mev_fm3": [
                    self.pressure_min_mev_fm3,
                    self.pressure_max_causal_mev_fm3,
                ],
            },
            "causal_domain_policy": {
                "retained_endpoint_mass_density_g_cm3": CAUSAL_MASS_DENSITY_MAX_G_CM3,
                "source_endpoint_baryon_density_fm3": CAUSAL_BARYON_DENSITY_MAX_FM3,
                "above_endpoint": "explicit_diagnostic_fit_evaluation_only_not_solver_use",
                "repair_or_clipping": "none",
            },
            "interpolation_method": "none_direct_Appendix_C_equations",
            "inversion_method": "Brent_bracket_in_log10_mass_density_no_extrapolation",
            "derivative_or_sound_speed_method": (
                "analytical_derivative_of_equation_C4; cs2=(P/epsilon)*dlogP/dlogrho"
            ),
            "phase_and_surface_metadata": {
                "unified_crust_core": True,
                "declared_internal_energy_density_jumps": [],
                "surface_type": "continuous_not_self_bound",
                "surface_energy_density_mev_fm3": 0.0,
                "stellar_termination": "P_at_rho_1e6_g_cm3_source_lower_boundary",
                "compose_phase_labels": {
                    "outer_crust": 1,
                    "inner_crust": 2,
                    "core": 0,
                },
            },
            "validation_status": self.validation_status,
            "validation_scope": {
                key: list(value) for key, value in self.validation_scope.items()
            },
        }


def make_bsk24_eos() -> BSk24AnalyticEos:
    """Return the approved causal-domain BSk24 production adapter."""
    return BSk24AnalyticEos()


__all__ = [
    "BSk24AnalyticEos",
    "BSk24DomainError",
    "BSk24InversionError",
    "CAUSAL_BARYON_DENSITY_MAX_FM3",
    "CAUSAL_MASS_DENSITY_MAX_G_CM3",
    "C_LIGHT_CM_S",
    "ENERGY_COEFFICIENTS",
    "FIT_MASS_DENSITY_MAX_G_CM3",
    "FIT_MASS_DENSITY_MIN_G_CM3",
    "MEV_FM3_TO_ERG_CM3",
    "MEV_FM3_TO_MASS_DENSITY_G_CM3",
    "MODEL_NAME",
    "MODEL_VERSION",
    "PRESSURE_COEFFICIENTS",
    "VALIDATION_SCOPE",
    "VALIDATION_STATUS",
    "make_bsk24_eos",
]
