"""
Vision and text analysis module using CLIP models.
"""

from .vit import VideoQuery, ImageQuery, MODELS, video_embeddings_path, image_embeddings_path

__all__ = ["VideoQuery", "ImageQuery", "MODELS", "video_embeddings_path", "image_embeddings_path"]