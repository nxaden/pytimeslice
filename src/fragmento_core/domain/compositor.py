from typing import List, Literal

import numpy as np
import numpy.typing as npt


Orientation = Literal["vertical", "horizontal"]


def _build_timeslice(
    height: int,
    width: int,
    num_slices: int,
    images: List[np.ndarray],
    frame_indices: npt.NDArray[np.int_],
    orientation: Orientation,
) -> np.ndarray:
    """Build a time-slice image for the given orientation."""
    output = np.zeros((height, width, 3), dtype=np.uint8)
    span = width if orientation == "vertical" else height
    edges = np.linspace(0, span, num_slices + 1).round().astype(int)

    for i in range(num_slices):
        start, end = edges[i], edges[i + 1]
        if end <= start:
            continue

        frame = images[frame_indices[i]]

        if orientation == "vertical":
            output[:, start:end, :] = frame[:, start:end, :]
        else:
            output[start:end, :, :] = frame[start:end, :, :]

    return output


def build_timeslice(
    images: List[np.ndarray],
    orientation: Orientation = "vertical",
    num_slices: int | None = None,
    reverse_time: bool = False,
) -> np.ndarray:
    """Build a composite image by taking sequential vertical or horizontal slices
    from an ordered list of RGB images.
    """
    if not images:
        raise ValueError("No images loaded.")

    first = images[0]
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("Expected RGB images.")

    height, width, num_channels = first.shape

    for img in images:
        if img.shape != (height, width, num_channels):
            raise ValueError(
                "All images must have the same dimensions after preprocessing."
            )

    if num_slices is None:
        num_slices = len(images)

    if num_slices < 1:
        raise ValueError("num_slices must be at least 1.")

    frame_indices: npt.NDArray[np.int_] = (
        np.linspace(0, len(images) - 1, num_slices).round().astype(int)
    )

    if reverse_time:
        frame_indices = frame_indices[::-1]

    return _build_timeslice(
        height, width, num_slices, images, frame_indices, orientation
    )
