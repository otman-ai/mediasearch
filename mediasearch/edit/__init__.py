"""
Video editing and censorship module.
"""

from .censor import CensorText, CensorObjects
from .helpers import (
    compression,
    add_audio,
    blur_region,
    get_video_duration,
    extract_audio,
    cut_video,
    remove_audio,
    remove_intervals,
)

__all__ = [
    "CensorText",
    "CensorObjects",
    "compression",
    "add_audio",
    "blur_region",
    "get_video_duration",
    "extract_audio",
    "cut_video",
    "remove_audio",
    "remove_intervals",
]