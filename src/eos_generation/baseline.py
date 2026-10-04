"""Baseline for controlled BSk24 / BSk25 experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq
from typing import Any


MODEL_NAME = "BSK24_ANALYTIC_PEARSON2018_CORR2019"


MODEL_VERSION = "pearson2018-corr2019-v1"


VALIDATION_STATUS = "pass"


VALIDATION_SCOPE = {
    "status_applies_to": (
        "declared_Pearson_Equation_C4_pressure_and_analytical_derivative_representation",
        "declared_Pearson_Equation_C1_anchor_use",
        "effective_first_law_thermodynamic_reconstruction",
        "stated_units_published_fit_and_governed_case_specific_causal_domains",
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


@dataclass(frozen=True)
class BaselineDefinition:
    """Source-pinned constants; selecting a model performs no scientific work."""

    matter_model: str
    pressure_coefficients: tuple[float, ...]
    energy_coefficients: tuple[float, ...]
    energy_low_exponent: float
    causal_mass_density_max_g_cm3: float
    causal_baryon_density_max_fm3: float
    standard_anchor_epsilon_mev_fm3: float
    retained_epsilon_max_mev_fm3: float
    core_entry_epsilon_mev_fm3: float
    core_entry_baryon_density_fm3: float
    muon_onset_baryon_density_fm3: float
    outer_inner_transition_epsilon_mev_fm3: float

    @property
    def model_name(self) -> str:
        return f"{self.matter_model.upper()}_ANALYTIC_PEARSON2018_CORR2019"


_BSK24_DEFINITION = BaselineDefinition(
    "bsk24", PRESSURE_COEFFICIENTS, ENERGY_COEFFICIENTS, 1.16667,
    CAUSAL_MASS_DENSITY_MAX_G_CM3, CAUSAL_BARYON_DENSITY_MAX_FM3,
    152.4912472062717, 1508.9793344234, 76.5591451931, 0.0807555,
    0.12577337, 0.253215574967,
)
_BSK25_DEFINITION = BaselineDefinition(
    "bsk25",
    (7.210, 5.196, 0.00328, 0.12516, 4.624, 12.16, 9.348, 1.6624,
     4.660, -28.232, 2.0638, 5.27, 14.365, 29.10, -2.130, 0.865,
     14.66, 0.069, 11.65, 6.30, -0.172, 14.18, 8.6),
    # C1 uses the existing helper's order (p13, p12, p11 at indices 10:13).
    # Paper Table C1 p8=2.54 governs; Ioffe 2023 uses 2.31 instead.
    (6.411e8, 8.76e10, 7.40e7, 6.31e6, 7.13e5, 22.11, 0.1217,
     2.54, 8317.0, 25.63, 3.92, 7.92, 2.507, 2.06),
    7.0 / 6.0, 3.81e15, 1.378,
    152.41651525155257, 3.81e15 / MEV_FM3_TO_MASS_DENSITY_G_CM3,
    81.14923220504647, 0.0855534, 0.12993982,
    0.2532200704511586,
)


def baseline_definition(matter_model: str = "bsk24") -> BaselineDefinition:
    if matter_model == "bsk24":
        return _BSK24_DEFINITION
    if matter_model == "bsk25":
        return _BSK25_DEFINITION
    raise ValueError("matter_model must be 'bsk24' or 'bsk25'")


class BSk24DomainError(ValueError):
    """Raised when an evaluation would leave the approved retained domain."""


class BSk24InversionError(RuntimeError):
    """Raised when a bracketed BSk24 inverse fails to converge."""


def _scalar_or_array(value: np.ndarray) -> float | np.ndarray:
    return float(value) if value.ndim == 0 else value


def _fermi(argument: np.ndarray) -> np.ndarray:
    """Stable evaluation of ``1 / (exp(argument) + 1)``."""
    return np.exp(-np.logaddexp(0.0, argument))


def _log10_pressure_from_xi(
    xi: np.ndarray, *, offset: float, coefficients=PRESSURE_COEFFICIENTS
) -> np.ndarray:
    a = coefficients
    denominator = 1.0 + a[3] * xi
    rational = (a[0] + a[1] * xi + a[2] * xi**3) / denominator
    result = rational * _fermi(a[4] * (xi - a[5]))
    result += (a[6] + a[7] * xi) * _fermi(a[8] * (a[5] - xi))
    result += (a[9] + a[10] * xi) * _fermi(a[11] * (a[12] - xi))
    result += (a[13] + a[14] * xi) * _fermi(a[15] * (a[16] - xi))
    result += a[17] / (1.0 + (a[19] * (xi - a[18])) ** 2)
    result += a[20] / (1.0 + (a[22] * (xi - a[21])) ** 2)
    return result + offset


def _dlog10_pressure_dxi(xi: np.ndarray, coefficients=PRESSURE_COEFFICIENTS) -> np.ndarray:
    """Analytical derivative of Appendix-C equation (C4)."""
    a = coefficients
    denominator = 1.0 + a[3] * xi
    numerator = a[0] + a[1] * xi + a[2] * xi**3
    numerator_prime = a[1] + 3.0 * a[2] * xi**2
    rational = numerator / denominator
    rational_prime = (numerator_prime * denominator - numerator * a[3]) / denominator**2

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
    coefficients=ENERGY_COEFFICIENTS,
    low_exponent: float = 1.16667,
) -> np.ndarray:
    """Pearson et al. Appendix-C equation (C1), BSk24 coefficients."""
    a = coefficients
    n = baryon_density_fm3
    high = (a[9] * n) ** a[12] / (1.0 + a[11] * n)
    high_weight = 1.0 / (1.0 + (a[10] * n) ** a[13])
    middle = a[5] * n ** a[6] * (1.0 + a[7] * n)
    middle_weight = 1.0 / (1.0 + a[8] * n)
    low = (a[0] * n) ** low_exponent / (1.0 + np.sqrt(a[1] * n))
    low *= (1.0 + np.sqrt(a[3] * n)) / (1.0 + np.sqrt(a[2] * n))
    low /= 1.0 + np.sqrt(a[4] * n)
    return (
        low * middle_weight
        + middle * (1.0 - middle_weight) * high_weight
        + high * (1.0 - high_weight)
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

    def __init__(self, matter_model: str = "bsk24"):
        self.definition = baseline_definition(matter_model)
        self.matter_model = matter_model
        self.model_name = self.definition.model_name

    def mass_density_from_energy_density(self, epsilon: Any) -> np.ndarray:
        return _mass_density_from_energy_density(
            epsilon, causal_mass_density_max_g_cm3=self.definition.causal_mass_density_max_g_cm3
        )

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
                coefficients=self.definition.pressure_coefficients,
            )
        )

    @property
    def pressure_max_causal_mev_fm3(self) -> float:
        return float(
            10.0
            ** _log10_pressure_from_xi(
                np.asarray(math.log10(self.definition.causal_mass_density_max_g_cm3)),
                offset=PRESSURE_LOG10_OFFSET_MEV_FM3,
                coefficients=self.definition.pressure_coefficients,
            )
        )

    @property
    def energy_density_min_mev_fm3(self) -> float:
        return FIT_MASS_DENSITY_MIN_G_CM3 / MEV_FM3_TO_MASS_DENSITY_G_CM3

    @property
    def energy_density_max_causal_mev_fm3(self) -> float:
        return self.definition.causal_mass_density_max_g_cm3 / MEV_FM3_TO_MASS_DENSITY_G_CM3

    @property
    def energy_density_max_published_fit_mev_fm3(self) -> float:
        """Upper total-energy-density bound of the published analytical fit."""

        return FIT_MASS_DENSITY_MAX_G_CM3 / MEV_FM3_TO_MASS_DENSITY_G_CM3

    def published_fit_pressure_from_mass_density(
        self,
        mass_density_g_cm3: Any,
    ) -> float | np.ndarray:
        """Evaluate Appendix-C equation (C4) on its published fit domain.

        This wider evaluator is not a causal repair and is not used by the
        direct BSk24 production barotrope.  It is available to governed
        deformed proposals whose *combined* sound speed remains causal above
        the direct-BSk24 endpoint.
        """

        rho = self._require_range(
            mass_density_g_cm3,
            name="published_fit_mass_density_g_cm3",
            lower=FIT_MASS_DENSITY_MIN_G_CM3,
            upper=FIT_MASS_DENSITY_MAX_G_CM3,
        )
        pressure = 10.0 ** _log10_pressure_from_xi(
            np.log10(rho),
            offset=PRESSURE_LOG10_OFFSET_MEV_FM3,
            coefficients=self.definition.pressure_coefficients,
        )
        return _scalar_or_array(pressure)

    def published_fit_pressure_from_energy_density(
        self,
        energy_density_mev_fm3: Any,
    ) -> float | np.ndarray:
        epsilon = self._require_range(
            energy_density_mev_fm3,
            name="published_fit_total_energy_density_mev_fm3",
            lower=self.energy_density_min_mev_fm3,
            upper=self.energy_density_max_published_fit_mev_fm3,
        )
        rho = epsilon * MEV_FM3_TO_MASS_DENSITY_G_CM3
        pressure = 10.0 ** _log10_pressure_from_xi(
            np.log10(rho),
            offset=PRESSURE_LOG10_OFFSET_MEV_FM3,
            coefficients=self.definition.pressure_coefficients,
        )
        return _scalar_or_array(pressure)

    def published_fit_sound_speed_squared_from_mass_density(
        self,
        mass_density_g_cm3: Any,
    ) -> float | np.ndarray:
        """Return the unmodified C4 derivative on the published fit domain."""

        rho = self._require_range(
            mass_density_g_cm3,
            name="published_fit_mass_density_g_cm3",
            lower=FIT_MASS_DENSITY_MIN_G_CM3,
            upper=FIT_MASS_DENSITY_MAX_G_CM3,
        )
        pressure = np.asarray(
            self.published_fit_pressure_from_mass_density(rho), dtype=float
        )
        epsilon = rho / MEV_FM3_TO_MASS_DENSITY_G_CM3
        result = pressure * _dlog10_pressure_dxi(np.log10(rho), self.definition.pressure_coefficients) / epsilon
        return _scalar_or_array(result)

    def pressure_from_mass_density(self, mass_density_g_cm3: Any) -> float | np.ndarray:
        rho = self._require_range(
            mass_density_g_cm3,
            name="mass_density_g_cm3",
            lower=FIT_MASS_DENSITY_MIN_G_CM3,
            upper=self.definition.causal_mass_density_max_g_cm3,
        )
        pressure = 10.0 ** _log10_pressure_from_xi(
            np.log10(rho),
            offset=PRESSURE_LOG10_OFFSET_MEV_FM3,
            coefficients=self.definition.pressure_coefficients,
        )
        return _scalar_or_array(pressure)

    def diagnostic_pressure_from_mass_density(
        self,
        mass_density_g_cm3: Any,
    ) -> float | np.ndarray:
        """Evaluate the published fit above causality only for explicit diagnostics."""
        return self.published_fit_pressure_from_mass_density(mass_density_g_cm3)

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
        pressure = 10.0 ** _log10_pressure_from_xi(np.log10(rho), offset=0.0, coefficients=self.definition.pressure_coefficients)
        return _scalar_or_array(pressure)

    def pressure_from_energy_density(
        self, energy_density_mev_fm3: Any
    ) -> float | np.ndarray:
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
            coefficients=self.definition.pressure_coefficients,
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
        upper = math.log10(self.definition.causal_mass_density_max_g_cm3)
        lower_pressure = self.pressure_min_mev_fm3
        upper_pressure = self.pressure_max_causal_mev_fm3
        for index, target in enumerate(flat):
            if target == lower_pressure:
                densities[index] = FIT_MASS_DENSITY_MIN_G_CM3
                continue
            if target == upper_pressure:
                densities[index] = self.definition.causal_mass_density_max_g_cm3
                continue
            log_target = math.log10(float(target))
            try:
                root = brentq(
                    lambda xi: float(
                        _log10_pressure_from_xi(
                            np.asarray(xi), offset=PRESSURE_LOG10_OFFSET_MEV_FM3,
                            coefficients=self.definition.pressure_coefficients
                        )
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
                    f"{self.matter_model.upper()} pressure inversion failed for {target!r} MeV/fm^3"
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
            upper=self.definition.causal_mass_density_max_g_cm3,
        )
        derivative = _dlog10_pressure_dxi(np.log10(rho), self.definition.pressure_coefficients)
        return _scalar_or_array(derivative)

    def sound_speed_squared_from_mass_density(
        self,
        mass_density_g_cm3: Any,
    ) -> float | np.ndarray:
        rho = self._require_range(
            mass_density_g_cm3,
            name="mass_density_g_cm3",
            lower=FIT_MASS_DENSITY_MIN_G_CM3,
            upper=self.definition.causal_mass_density_max_g_cm3,
        )
        pressure = np.asarray(self.pressure_from_mass_density(rho), dtype=float)
        epsilon = rho / MEV_FM3_TO_MASS_DENSITY_G_CM3
        result = pressure * _dlog10_pressure_dxi(np.log10(rho), self.definition.pressure_coefficients) / epsilon
        return _scalar_or_array(result)

    def sound_speed_squared_from_pressure(
        self, pressure_mev_fm3: Any
    ) -> float | np.ndarray:
        rho = self.mass_density_from_pressure(pressure_mev_fm3)
        return self.sound_speed_squared_from_mass_density(rho)

    def _mass_density_from_baryon_density(self, baryon_density_fm3: float) -> float:
        n = np.asarray(float(baryon_density_fm3))
        energy = _energy_per_baryon_above_iron_ground_mev(
            n, self.definition.energy_coefficients, self.definition.energy_low_exponent
        )
        total_per_baryon = NEUTRON_REST_ENERGY_MEV + IRON56_GROUND_OFFSET_MEV + energy
        return float(n * total_per_baryon * MEV_FM3_TO_MASS_DENSITY_G_CM3)

    def baryon_density_from_mass_density(
        self, mass_density_g_cm3: Any
    ) -> float | np.ndarray:
        rho = self._require_range(
            mass_density_g_cm3,
            name="mass_density_g_cm3",
            lower=FIT_MASS_DENSITY_MIN_G_CM3,
            upper=self.definition.causal_mass_density_max_g_cm3,
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
                    f"{self.matter_model.upper()} baryon-density inversion failed for {target!r} g/cm^3"
                ) from exc
        return _scalar_or_array(result.reshape(rho.shape))

    def source_state_from_mass_density(self, mass_density_g_cm3: Any) -> dict[str, Any]:
        rho = self._require_range(
            mass_density_g_cm3,
            name="mass_density_g_cm3",
            lower=FIT_MASS_DENSITY_MIN_G_CM3,
            upper=self.definition.causal_mass_density_max_g_cm3,
        )
        n = np.asarray(self.baryon_density_from_mass_density(rho), dtype=float)
        energy_above_ground = _energy_per_baryon_above_iron_ground_mev(
            n, self.definition.energy_coefficients, self.definition.energy_low_exponent
        )
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
        cs2 = float(self.sound_speed_squared_from_mass_density(mass_density))
        return epsilon, cs2

    def provenance(self) -> dict[str, Any]:
        """Return complete JSON-safe model identity and scientific conventions."""
        result = {
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
                    self.definition.causal_mass_density_max_g_cm3,
                ],
                "governed_deformation_assessment_mass_density_g_cm3": [
                    FIT_MASS_DENSITY_MIN_G_CM3,
                    FIT_MASS_DENSITY_MAX_G_CM3,
                ],
                "production_retained_pressure_mev_fm3": [
                    self.pressure_min_mev_fm3,
                    self.pressure_max_causal_mev_fm3,
                ],
            },
            "causal_domain_policy": {
                "retained_endpoint_mass_density_g_cm3": self.definition.causal_mass_density_max_g_cm3,
                "source_endpoint_baryon_density_fm3": self.definition.causal_baryon_density_max_fm3,
                "above_direct_endpoint": (
                    "direct_BSk24_solver_use_forbidden; governed_nonzero_deformations_may_use_"
                    "the_published_fit_until_the_combined_first_causal_crossing"
                ),
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

        if self.matter_model == "bsk25":
            import json
            from pathlib import Path

            source = json.loads((Path(__file__).parent / "source_manifest.json").read_text(encoding="utf-8"))["additional_models"]["bsk25"]
            result["source_artifact_or_routine"] = {
                "independent_implementation_oracle": "Ioffe bskfit18.f revision 2023-02-13, KEOS=25; C4 only",
                "underlying_tabulated_oracle": "CompOSE PCP(BSK25), EoS ID 257",
            }
            result["source_checksums_sha256"]["compose_eos_zip"] = source["underlying_tabulated_oracle"]["sha256"]
            result["retrieval_date"] = source["retrieval_date"]
            result["source_discrepancies"] = source["source_discrepancies"]
            result["causal_domain_policy"]["above_direct_endpoint"] = result["causal_domain_policy"]["above_direct_endpoint"].replace("BSk24", "BSk25")
        return result


def make_bsk24_eos() -> BSk24AnalyticEos:
    """Return the approved causal-domain BSk24 production adapter."""
    return BSk24AnalyticEos()


class BSk25AnalyticEos(BSk24AnalyticEos):
    """Pearson's BSk25 fit with source-pinned model-specific domains."""

    def __init__(self):
        super().__init__("bsk25")


def make_baseline_eos(matter_model: str = "bsk24") -> BSk24AnalyticEos:
    baseline_definition(matter_model)
    return BSk24AnalyticEos() if matter_model == "bsk24" else BSk25AnalyticEos()


COMPOSE_CORE_ENTRY_EPSILON_MEV_FM3 = 76.5591451931


def _mass_density_from_energy_density(
    epsilon: np.ndarray, *, causal_mass_density_max_g_cm3=CAUSAL_MASS_DENSITY_MAX_G_CM3
) -> np.ndarray:
    """Convert units while snapping only roundoff-adjacent declared endpoints."""
    rho = np.asarray(epsilon, dtype=float) * MEV_FM3_TO_MASS_DENSITY_G_CM3
    guard = 16.0 * np.finfo(float).eps
    rho = np.where(
        np.isclose(rho, FIT_MASS_DENSITY_MIN_G_CM3, rtol=guard, atol=0.0),
        FIT_MASS_DENSITY_MIN_G_CM3,
        rho,
    )
    rho = np.where(
        np.isclose(rho, causal_mass_density_max_g_cm3, rtol=guard, atol=0.0),
        causal_mass_density_max_g_cm3,
        rho,
    )
    rho = np.where(
        np.isclose(rho, FIT_MASS_DENSITY_MAX_G_CM3, rtol=guard, atol=0.0),
        FIT_MASS_DENSITY_MAX_G_CM3,
        rho,
    )
    return rho
