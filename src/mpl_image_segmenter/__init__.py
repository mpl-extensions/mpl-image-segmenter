"""Manually segment images with matplotlib."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mpl-image-segmenter")
except PackageNotFoundError:
    __version__ = "uninstalled"

from ._segmenter import image_segmenter

__author__ = "Ian Hunt-Isaak"
__email__ = "ianhuntisaak@gmail.com"
__all__ = [
    "image_segmenter",
]
