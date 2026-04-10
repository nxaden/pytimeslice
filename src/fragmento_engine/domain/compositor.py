from __future__ import annotations

from typing import Sequence

import numpy as np

from .models import CompositeResult, RGBImage, TimeslicePlan, TimesliceSpec
from .planner import build_timeslice_plan


def _validate_images(images: Sequence[RGBImage]) -> tuple[int, int, int]:
    if not images:
        raise ValueError("No images loaded.")

    first = images[0]
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("Expected RGB images.")

    height, width, channels = first.shape

    for i, img in enumerate(images):
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Image at index {i} is not an RGB image.")
        if img.shape != (height, width, channels):
            raise ValueError(
                "All images must have the same dimensions after preprocessing."
            )

    return height, width, channels


def apply_timeslice_plan(
    images: Sequence[RGBImage],
    plan: TimeslicePlan,
) -> CompositeResult:
    height, width, _ = _validate_images(images)
    output = np.zeros((height, width, 3), dtype=np.uint8)

    for band in plan.bands:
        frame = images[band.frame_index]

        if plan.orientation == "vertical":
            output[:, band.start : band.end, :] = frame[:, band.start : band.end, :]
        else:
            output[band.start : band.end, :, :] = frame[band.start : band.end, :, :]

    used_frame_indices = sorted({band.frame_index for band in plan.bands})

    return CompositeResult(
        image=output,
        plan=plan,
        used_frame_indices=used_frame_indices,
    )


def build_timeslice(
    images: Sequence[RGBImage],
    spec: TimesliceSpec | None = None,
) -> CompositeResult:
    if spec is None:
        spec = TimesliceSpec()

    plan = build_timeslice_plan(images=images, spec=spec)
    return apply_timeslice_plan(images=images, plan=plan)
