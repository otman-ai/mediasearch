import pytest

def test_search_images(timer):
    from mediasearch.vit import ImageQuery
    import os
    # remove existing test cash
    default = os.path.join(os.path.expanduser("~"), ".cache")
    cash_dir = os.path.join(os.getenv("XDG_CACHE_HOME", default), "mediasearch")
    cash_file_path = os.path.join(cash_dir, "test_image_embeddings.h5")
    if os.path.exists(cash_file_path):
        os.remove(cash_file_path)
    config = {
    "model_name": "ViT-B/32",
    "threshold": 0.25,
    "cash": cash_file_path,
    "debug": False
}


    image_search = ImageQuery(**config)
    images = ["assets/frame.jpg", "assets/frame1.jpg", "assets/frame2.JPG"]

    with timer("insert_images"):
        image_search.insert_images(images=images)

    solution =  {'assets/frame.jpg': 0.26540297269821167, 'assets/frame2.JPG': 0.2592819333076477}

    with timer("search"):
        results = image_search.search("buggy")

    assert results.keys() == solution.keys()
    assert list(results.values()) == list(solution.values())
    os.remove(cash_file_path)

def test_search_videos(timer):
    from mediasearch.vit import VideoQuery
    import os
    # remove existing test cash
    default = os.path.join(os.path.expanduser("~"), ".cache")
    cash_dir = os.path.join(os.getenv("XDG_CACHE_HOME", default), "mediasearch")
    cash_file_path = os.path.join(cash_dir, "test_embeddings.h5")
    if os.path.exists(cash_file_path):
        os.remove(cash_file_path)
    config = {
        "model_name": "ViT-B/32",
        "frame_rate": 3,
        "threshold": 0.25,
        "cash": cash_file_path,
        "debug": False
    }
    video_search = VideoQuery(**config)
    videos = [
    "assets/video0.mp4",
    "assets/video1.mp4",
    "assets/video2.mp4",
    "assets/video3.mp4",
    "assets/video4.mp4",
    "assets/video5.mp4"
    ]
    query = "car"

    with timer("insert_videos"):
        video_search.insert_videos(videos_path=videos)

    solution = {'assets/video0.mp4': [(0.0, 1.0, 0.27001953125), (1.0, 2.0, 0.274658203125), (2.0, 3.0, 0.27099609375), (6.0, 7.0, 0.251220703125), (8.0, 9.0, 0.26416015625), (9.0, 10.0, 0.267822265625), (10.0, 11.0, 0.254638671875), (11.0, 12.0, 0.2548828125), (12.0, 12.666666666666666, 0.274658203125)], 'assets/video2.mp4': [(0.0, 1.0, 0.255859375)]}
    with timer("search"):
        results = video_search.search(query)

    assert results.keys() == solution.keys()
    assert list(results.values()) == list(solution.values())
    os.remove(cash_file_path)
