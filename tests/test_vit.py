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
    image_search = ImageQuery(cash=cash_file_path)
    image_search.insert_images(images=["assets/frame.jpg", "assets/frame2.JPG"])
    solution = {'assets/frame.jpg': 26.54030418395996, 'assets/frame2.JPG': 25.928192138671875}
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
    video_search = VideoQuery(cash=cash_file_path, threshold=0.05)
    video_search.insert_videos(videos_path=["assets/video.mp4"])
    solution = {'assets/video.mp4':  [(6.0, 7.0), (9.0, 10.0), (14.0, 15.0), (26.0, 27.0), (27.0, 28.0), (28.0, 29.0)]}
    results = video_search.search("two black guys")
    assert results.keys() == solution.keys()
    assert list(results.values()) == list(solution.values())
    os.remove(cash_file_path)
