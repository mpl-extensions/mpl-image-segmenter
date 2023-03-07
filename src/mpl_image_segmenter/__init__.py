"""Manually segment images with matplotlib."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mpl-image-segmenter")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Ian Hunt-Isaak"
__email__ = "ianhuntisaak@gmail.com"
