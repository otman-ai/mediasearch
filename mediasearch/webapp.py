"""
Gradio web app for MediaSearch.

A small browser UI around the core library: search a corpus of videos/images
by text query, index new media, censor objects, and run the basic video edits.

Run it with::

    mediasearch app            # or: mediasearch-app
    python -m mediasearch.webapp

Requires the optional ``app`` extra::

    pip install "mediasearch[app]"
"""

import os
import tempfile
import threading
from typing import List, Optional

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - only hit without the extra
    raise ImportError(
        "The Gradio web app needs the optional 'app' dependencies. "
        'Install them with: pip install "mediasearch[app]"'
    ) from exc

from mediasearch.edit import (
    CensorObjects,
    compression,
    cut_video,
    extract_audio,
    get_video_duration,
    remove_audio,
)
from mediasearch.vit import (
    MODELS,
    ImageQuery,
    VideoQuery,
    image_embeddings_path,
    video_embeddings_path,
)

VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm")
IMAGE_EXTS = (".png", ".jpg", ".jpeg")
CENSOR_LABELS = ["faces", "license_plates"]

_lock = threading.Lock()
_video_query: Optional[VideoQuery] = None
_image_query: Optional[ImageQuery] = None
_censor_objects: dict = {}

_MODEL = os.getenv("MEDIASEARCH_MODEL", "ViT-B/32")
_FRAME_RATE = int(os.getenv("MEDIASEARCH_FRAME_RATE", "10"))
_THRESHOLD = float(os.getenv("MEDIASEARCH_THRESHOLD", "0.25"))
_VIDEO_CASH = os.getenv("MEDIASEARCH_VIDEO_CASH", video_embeddings_path)
_IMAGE_CASH = os.getenv("MEDIASEARCH_IMAGE_CASH", image_embeddings_path)
_OUTPUT_DIR = os.getenv(
    "MEDIASEARCH_OUTPUT_DIR", os.path.join(tempfile.gettempdir(), "mediasearch-app")
)


def get_video_query() -> VideoQuery:
    """CLIP model loading is expensive, so keep one instance around."""
    global _video_query
    if _video_query is None:
        with _lock:
            if _video_query is None:
                _video_query = VideoQuery(
                    model_name=_MODEL,
                    frame_rate=_FRAME_RATE,
                    threshold=_THRESHOLD,
                    cash=_VIDEO_CASH,
                )
    return _video_query


def get_image_query() -> ImageQuery:
    global _image_query
    if _image_query is None:
        with _lock:
            if _image_query is None:
                _image_query = ImageQuery(
                    model_name=_MODEL,
                    threshold=_THRESHOLD,
                    cash=_IMAGE_CASH,
                )
    return _image_query


def get_censor_objects(labels: List[str]) -> CensorObjects:
    key = tuple(sorted(labels))
    if key not in _censor_objects:
        with _lock:
            if key not in _censor_objects:
                _censor_objects[key] = CensorObjects(labels=list(key))
    return _censor_objects[key]


def _output_path(suffix: str) -> str:
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=suffix, dir=_OUTPUT_DIR)
    os.close(fd)
    return path


def _paths(files) -> List[str]:
    if not files:
        return []
    out = []
    for f in files:
        out.append(f if isinstance(f, str) else f.name)
    return out


# --------------------------------------------------------------------- handlers
def search_videos(query: str, threshold: float, united: bool):
    if not query or not query.strip():
        raise gr.Error("Enter a search query.")
    vq = get_video_query()
    vq.threshold = threshold
    results = vq.search(query.strip(), is_united_timestamp=united) or {}
    rows = [
        [os.path.basename(video), video, round(start, 2), round(end, 2), round(score, 4)]
        for video, hits in results.items()
        for start, end, score in hits
    ]
    rows.sort(key=lambda r: r[4], reverse=True)
    if not rows:
        gr.Info("No matching segments.")
    return rows


def insert_videos(files):
    paths = _paths(files)
    if not paths:
        raise gr.Error("Upload one or more videos first.")
    bad = [p for p in paths if not p.lower().endswith(VIDEO_EXTS)]
    if bad:
        raise gr.Error(f"Unsupported video file(s): {', '.join(map(os.path.basename, bad))}")
    get_video_query().insert_videos(videos_path=paths)
    return f"Indexed {len(paths)} video(s)."


def search_images(query: str, threshold: float):
    if not query or not query.strip():
        raise gr.Error("Enter a search query.")
    iq = get_image_query()
    iq.threshold = threshold
    results = iq.search(query.strip()) or {}
    rows = sorted(
        ([os.path.basename(img), img, round(score, 4)] for img, score in results.items()),
        key=lambda r: r[2],
        reverse=True,
    )
    if not rows:
        gr.Info("No matching images.")
    return rows


def insert_images(files):
    paths = _paths(files)
    if not paths:
        raise gr.Error("Upload one or more images first.")
    bad = [p for p in paths if not p.lower().endswith(IMAGE_EXTS)]
    if bad:
        raise gr.Error(f"Unsupported image file(s): {', '.join(map(os.path.basename, bad))}")
    get_image_query().insert_images(images=paths)
    return f"Indexed {len(paths)} image(s)."


def censor(file, labels: List[str]):
    if not file:
        raise gr.Error("Upload a video or image.")
    if not labels:
        raise gr.Error("Pick at least one label to censor.")
    src = file if isinstance(file, str) else file.name
    ext = os.path.splitext(src)[1].lower()
    worker = get_censor_objects(labels)
    if ext in VIDEO_EXTS:
        out = _output_path(ext)
        worker.censor_video(src, out)
    elif ext in IMAGE_EXTS:
        out = _output_path(ext)
        worker.censor_image(src, out)
    else:
        raise gr.Error(f"Unsupported file type: {ext}")
    return out


def edit(file, operation: str, start_time: float, duration: float):
    if not file:
        raise gr.Error("Upload a video.")
    src = file if isinstance(file, str) else file.name
    if not src.lower().endswith(VIDEO_EXTS):
        raise gr.Error("Upload a video file.")

    if operation == "Cut":
        out = _output_path(".mp4")
        cut_video(src, out, start_time=start_time, duration=duration)
    elif operation == "Compress":
        out = _output_path(".mp4")
        compression(src, out)
    elif operation == "Extract audio":
        out = _output_path(".mp3")
        extract_audio(src, out)
    elif operation == "Remove audio":
        out = _output_path(".mp4")
        remove_audio(src, out)
    else:  # pragma: no cover - guarded by the dropdown choices
        raise gr.Error(f"Unknown operation: {operation}")
    return out


# ------------------------------------------------------------------------- app
def build_app() -> "gr.Blocks":
    with gr.Blocks(title="MediaSearch") as demo:
        gr.Markdown("# MediaSearch\nAI-powered media search and editing toolkit.")

        with gr.Tab("Video search"):
            with gr.Row():
                v_query = gr.Textbox(label="Query", scale=4, placeholder="dogs running")
                v_threshold = gr.Slider(0.0, 1.0, value=_THRESHOLD, step=0.01, label="Threshold")
                v_united = gr.Checkbox(value=True, label="Merge adjacent timestamps")
            v_btn = gr.Button("Search", variant="primary")
            v_results = gr.Dataframe(
                headers=["video", "path", "start (s)", "end (s)", "score"],
                label="Matching segments",
                interactive=False,
                wrap=True,
            )
            v_btn.click(search_videos, [v_query, v_threshold, v_united], v_results)
            v_query.submit(search_videos, [v_query, v_threshold, v_united], v_results)

            gr.Markdown("### Index videos")
            v_upload = gr.File(file_count="multiple", file_types=list(VIDEO_EXTS), label="Videos")
            v_insert_btn = gr.Button("Index videos")
            v_insert_status = gr.Markdown()
            v_insert_btn.click(insert_videos, v_upload, v_insert_status)

        with gr.Tab("Image search"):
            with gr.Row():
                i_query = gr.Textbox(label="Query", scale=4, placeholder="a red car")
                i_threshold = gr.Slider(0.0, 1.0, value=_THRESHOLD, step=0.01, label="Threshold")
            i_btn = gr.Button("Search", variant="primary")
            i_results = gr.Dataframe(
                headers=["image", "path", "score"],
                label="Ranked images",
                interactive=False,
                wrap=True,
            )
            i_btn.click(search_images, [i_query, i_threshold], i_results)
            i_query.submit(search_images, [i_query, i_threshold], i_results)

            gr.Markdown("### Index images")
            i_upload = gr.File(file_count="multiple", file_types=list(IMAGE_EXTS), label="Images")
            i_insert_btn = gr.Button("Index images")
            i_insert_status = gr.Markdown()
            i_insert_btn.click(insert_images, i_upload, i_insert_status)

        with gr.Tab("Censor objects"):
            c_file = gr.File(label="Video or image", file_types=list(VIDEO_EXTS + IMAGE_EXTS))
            c_labels = gr.CheckboxGroup(CENSOR_LABELS, value=["faces"], label="Blur")
            c_btn = gr.Button("Censor", variant="primary")
            c_out = gr.File(label="Result")
            c_btn.click(censor, [c_file, c_labels], c_out)

        with gr.Tab("Edit"):
            e_file = gr.File(label="Video", file_types=list(VIDEO_EXTS))
            e_op = gr.Dropdown(
                ["Cut", "Compress", "Extract audio", "Remove audio"],
                value="Cut",
                label="Operation",
            )
            with gr.Row():
                e_start = gr.Number(value=0, label="Start time (s) — Cut only")
                e_duration = gr.Number(value=10, label="Duration (s) — Cut only")
            e_btn = gr.Button("Run", variant="primary")
            e_out = gr.File(label="Result")
            e_btn.click(edit, [e_file, e_op, e_start, e_duration], e_out)

        gr.Markdown(
            f"Model `{_MODEL}` · frame rate {_FRAME_RATE} · "
            f"video index `{_VIDEO_CASH}` · image index `{_IMAGE_CASH}`"
        )
    return demo


def main() -> None:
    """Entry point for the ``mediasearch app`` / ``mediasearch-app`` command."""
    demo = build_app()
    demo.queue().launch(
        server_name=os.getenv("MEDIASEARCH_HOST", "127.0.0.1"),
        server_port=int(os.getenv("MEDIASEARCH_PORT", os.getenv("PORT", "7860"))),
        share=os.getenv("MEDIASEARCH_SHARE", "0") == "1",
    )


if __name__ == "__main__":
    main()
