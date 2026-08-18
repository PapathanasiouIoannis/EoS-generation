"""Thin ordinary-experiment runtime assembled from private components."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from eos_generation._internal.artifacts import (
    ensure_within_runs,
    project_root,
    repository_root_scope,
)
from eos_generation._internal.execution import RunCallbacks, run_bsk24_trial
from eos_generation._internal.loading import load_bsk24_trial
from eos_generation._internal.planning import (
    BSk24TrialConfig,
    BSk24TrialPlan,
    prepare_bsk24_trial,
)
from eos_generation._internal.provenance import _source_hashes
from eos_generation._internal.stellar import _run_stellar
from eos_generation._internal.thermodynamics import (
    _deformations,
    _raw_gate_frame,
    _thermodynamic_convergence,
    _thermodynamic_profile_frame,
    _thermodynamic_residual_frame,
)
from eos_generation.bsk24.deformation import (
    build_windowed_eos,
    raw_local_physics_gate,
    window_characterization,
)
from eos_generation.bsk24.reconstruction import build_consistent_baseline
from eos_generation.reporting.plot_orchestration import (
    generate_bsk24_trial_plots,
)
from eos_generation.reporting.validation import (
    validate_bsk24_trial_packet_layers,
)


@dataclass
class TrialResult:
    """Passive handle to one saved and validated trial packet."""

    packet_path: Path
    config: BSk24TrialConfig
    metadata: dict[str, Any]
    plot_inventory: pd.DataFrame
    repository_root: Path | None = None

    def __post_init__(self) -> None:
        root = (
            project_root()
            if self.repository_root is None
            else Path(self.repository_root)
        ).expanduser().resolve(strict=False)
        with repository_root_scope(root):
            packet = ensure_within_runs(self.packet_path)
        self.repository_root = root
        self.packet_path = packet

    @property
    def accepted_cases(self) -> pd.DataFrame:
        return self._ledger_rows("accepted")

    @property
    def rejected_cases(self) -> pd.DataFrame:
        return self._ledger_rows("rejected")

    @property
    def identity_status(self) -> str:
        return str(self.metadata.get("identity_status", "unavailable"))

    @property
    def convergence_status(self) -> str:
        return str(
            self.metadata.get("numerical_convergence_status", "unavailable")
        )

    @property
    def figures(self) -> tuple[Path, ...]:
        rows = self.plot_inventory.loc[
            self.plot_inventory["status"].isin(
                ("generated", "generated_partial")
            )
        ]
        return tuple(self._packet_artifact(value) for value in rows["relative_path"])

    def _packet_artifact(self, relative_path: Any) -> Path:
        if not isinstance(relative_path, (str, Path)):
            raise TypeError("packet artifact path must be text or a Path")
        relative = Path(relative_path)
        if relative.is_absolute():
            raise ValueError(f"packet artifact path must be relative: {relative_path!r}")
        with repository_root_scope(self.repository_root):
            packet = ensure_within_runs(self.packet_path)
            candidate = (packet / relative).resolve(strict=False)
            try:
                candidate.relative_to(packet)
            except ValueError as exc:
                raise ValueError(
                    f"packet artifact path escapes its packet: {relative_path!r}"
                ) from exc
            return ensure_within_runs(candidate)

    def _ledger_rows(self, status: str) -> pd.DataFrame:
        with repository_root_scope(self.repository_root):
            path = ensure_within_runs(self.packet_path / "case_ledger.csv")
            if not path.is_file():
                return pd.DataFrame()
            frame = pd.read_csv(path)
        return frame.loc[frame["status"] == status].reset_index(drop=True)

    def table(self, relative_path: str) -> pd.DataFrame:
        with repository_root_scope(self.repository_root):
            path = ensure_within_runs(self._packet_artifact(relative_path))
            if not path.is_file():
                raise FileNotFoundError(path)
            return pd.read_csv(path)

    def figure_inventory(self) -> pd.DataFrame:
        return self.plot_inventory.copy()

    def show_gallery(self) -> tuple[Path, ...]:
        paths = self.figures
        try:
            from IPython.display import Image, display
        except ImportError:
            return paths
        for path in paths:
            display(Image(filename=str(path)))
        skipped = self.plot_inventory.loc[
            self.plot_inventory["status"] == "skipped"
        ]
        if not skipped.empty:
            display(skipped[["figure", "reason"]])
        partial = self.plot_inventory.loc[
            self.plot_inventory["status"] == "generated_partial"
        ]
        if not partial.empty:
            display(partial[["figure", "status", "reason"]])
        return paths

    def regenerate_plots(
        self,
        groups: Sequence[str] = ("all-applicable",),
        *,
        authorize_plot_overwrite: bool = False,
    ) -> "TrialResult":
        generate_trial_plots(
            self.packet_path,
            groups=groups,
            authorize_plot_overwrite=authorize_plot_overwrite,
            repository_root=self.repository_root,
        )
        return load_trial(
            self.packet_path,
            repository_root=self.repository_root,
        )


def plan_trial(config: BSk24TrialConfig) -> BSk24TrialPlan:
    """Passively validate and estimate one ordinary trial."""
    return prepare_bsk24_trial(config)


def execute_trial(config: BSk24TrialConfig) -> TrialResult:
    """Explicitly execute one ordinary trial."""
    callbacks = RunCallbacks(
        prepare_trial=prepare_bsk24_trial,
        load_trial=load_trial,
        generate_plots=generate_trial_plots,
        validate_packet=validate_trial,
        build_consistent_baseline=build_consistent_baseline,
        raw_local_physics_gate=raw_local_physics_gate,
        raw_gate_frame=_raw_gate_frame,
        build_windowed_eos=build_windowed_eos,
        thermodynamic_profile_frame=_thermodynamic_profile_frame,
        thermodynamic_residual_frame=_thermodynamic_residual_frame,
        window_characterization=window_characterization,
        thermodynamic_convergence=_thermodynamic_convergence,
        run_stellar=_run_stellar,
    )
    return run_bsk24_trial(config, callbacks=callbacks)


def load_trial(
    packet_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> TrialResult:
    """Open an existing packet without scientific execution."""
    scope = (
        nullcontext()
        if repository_root is None
        else repository_root_scope(repository_root)
    )
    with scope:
        return load_bsk24_trial(packet_path, result_factory=TrialResult)


def validate_trial(
    packet_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Read-only validation of packet integrity and source equivalence."""
    scope = (
        nullcontext()
        if repository_root is None
        else repository_root_scope(repository_root)
    )
    with scope:
        return validate_bsk24_trial_packet_layers(
            packet_path,
            current_source_hashes=_source_hashes(),
            configuration_hash_fn=lambda payload: BSk24TrialConfig.from_dict(
                payload
            ).deterministic_hash(),
            required_source_paths=(
                "src/eos_generation/stellar/discontinuities.py",
                "src/eos_generation/stellar/tov.py",
                "src/eos_generation/_internal/sequence_tables.py",
            ),
        )


def generate_trial_plots(
    packet_path: str | Path,
    *,
    groups: Sequence[str] = ("all-applicable",),
    authorize_plot_overwrite: bool = False,
    _initial_packet_generation: bool = False,
    repository_root: str | Path | None = None,
) -> pd.DataFrame:
    """Generate figures strictly from already-saved packet tables."""
    scope = (
        nullcontext()
        if repository_root is None
        else repository_root_scope(repository_root)
    )
    with scope:
        if not _initial_packet_generation:
            validation = validate_trial(packet_path)
            if validation.get("status") != "pass":
                failures = validation.get("failures", [])
                detail = failures[0] if failures else "unspecified validation failure"
                raise ValueError(
                    "refusing to regenerate plots from an untrusted packet: "
                    f"{detail}"
                )
        return generate_bsk24_trial_plots(
            packet_path,
            groups=groups,
            authorize_plot_overwrite=authorize_plot_overwrite,
            _initial_packet_generation=_initial_packet_generation,
        )


__all__ = [
    "TrialResult",
    "execute_trial",
    "generate_trial_plots",
    "load_trial",
    "plan_trial",
    "validate_trial",
]
