from __future__ import annotations

import argparse
from pathlib import Path

from fragmento_engine.application.services import (
    RenderRequest,
    RenderTimesliceService,
)
from fragmento_engine.domain.models import TimesliceSpec
from fragmento_engine.infrastructure.image_loader import PILImageSequenceLoader
from fragmento_engine.infrastructure.image_writer import PILImageWriter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a time-slice image from a sequence of photos."
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Folder containing sequential images.",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path for the output image.",
    )
    parser.add_argument(
        "--orientation",
        choices=["vertical", "horizontal"],
        default="vertical",
        help="Use vertical strips or horizontal strips in the final composite.",
    )
    parser.add_argument(
        "--slices",
        type=int,
        default=None,
        help="Number of slices in the final image. Default: number of input images.",
    )
    parser.add_argument(
        "--resize-mode",
        choices=["crop", "resize"],
        default="crop",
        help="How to handle source images with different sizes.",
    )
    parser.add_argument(
        "--reverse-time",
        action="store_true",
        help="Reverse the time direction in the final image.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    spec = TimesliceSpec(
        orientation=args.orientation,
        num_slices=args.slices,
        reverse_time=args.reverse_time,
    )

    request = RenderRequest(
        input_folder=args.input_folder,
        spec=spec,
        resize_mode=args.resize_mode,
    )

    service = RenderTimesliceService(
        sequence_loader=PILImageSequenceLoader(),
        image_writer=PILImageWriter(),
    )

    response = service.render_to_file(
        request=request,
        output_file=args.output_file,
    )

    print(f"Rendered using {len(response.input_paths)} images.")
    print(f"Saved: {args.output_file}")


if __name__ == "__main__":
    main()
