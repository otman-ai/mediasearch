"""
MediaSearch - AI-powered media search and editing toolkit.

This package provides tools for:
- Video content search and analysis
- Image similarity matching
- Video editing and censorship
- Audio manipulation
"""

from .vit import VideoQuery, ImageQuery, MODELS
from .edit import (
    CensorText,
    CensorObjects,
    compression,
    add_audio,
    blur_region,
    get_video_duration,
    extract_audio,
    cut_video,
    remove_audio,
    remove_intervals,
)

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    # Video and Image Analysis
    "VideoQuery",
    "ImageQuery",
    "MODELS",
    # Video Editing
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