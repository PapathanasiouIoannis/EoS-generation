"""Deterministic bounded panel selection for dense saved-table figures."""

from __future__ import annotations

import math
from typing import Iterable


MAX_DENSE_PANELS = 6


def bounded_representative_values(
    values: Iterable[float],
    *,
    preferred: Iterable[float] = (),
    maximum: int = MAX_DENSE_PANELS,
) -> tuple[tuple[float, ...], int]:
    """Return endpoints, preferred values, and evenly spaced representatives.

    This is a presentation-only selection. Scientific tables and packet
    validation retain every value. The function is deterministic and never
    presents more than ``maximum`` panels.
    """

    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 2:
        raise ValueError("maximum panel count must be an integer of at least 2")
    ordered = tuple(
        sorted({float(value) for value in values if math.isfinite(float(value))})
    )
    if len(ordered) <= maximum:
        return ordered, 0

    chosen: set[float] = {ordered[0], ordered[-1]}
    for requested in preferred:
        requested = float(requested)
        if not math.isfinite(requested):
            continue
        closest = min(ordered, key=lambda value: (abs(value - requested), value))
        chosen.add(closest)
        if len(chosen) == maximum:
            break

    for slot in range(1, maximum):
        index = round(slot * (len(ordered) - 1) / maximum)
        chosen.add(ordered[index])
        if len(chosen) >= maximum:
            break
    if len(chosen) < maximum:
        for value in ordered:
            chosen.add(value)
            if len(chosen) == maximum:
                break

    selected = tuple(sorted(chosen))
    return selected, len(ordered) - len(selected)


__all__ = ["MAX_DENSE_PANELS", "bounded_representative_values"]
