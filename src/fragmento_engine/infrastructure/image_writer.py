from pathlib import Path

from PIL import Image

from fragmento_engine.domain.models import RGBImage


class PILImageWriter:
    """PIL-based infrastructure adapter for saving rendered images."""

    def save(self, image: RGBImage, output_file: Path) -> None:
        """Save an RGB numpy array to disk."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(output_file)
