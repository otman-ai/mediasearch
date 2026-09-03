import pytest

def test_search_images():
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
    image_search.insert_images(images=["assets/frame.jpg", "assets/frame2.JPG"])
    solution =  {'assets/frame.jpg': 0.26540300250053406,  'assets/frame2.JPG': 0.2592819333076477}
    results = image_search.search("buggy")
    assert results.keys() == solution.keys()
    assert list(results.values()) == list(solution.values())
    os.remove(cash_file_path)

def test_search_videos():
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
        "frame_rate": 10,
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
    
    video_search.insert_videos(videos_path=videos)
    solution = {'assets/video0.mp4': [(0.0, 3.3333333333333335), (6.666666666666667, 10.0), (10.0, 12.666666666666666)], 'assets/video2.mp4': [(0.0, 3.3333333333333335)]}
    results = video_search.search(query)
    assert results.keys() == solution.keys()
    assert list(results.values()) == list(solution.values())
    os.remove(cash_file_path)
