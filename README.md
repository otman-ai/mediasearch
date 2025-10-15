# Media Search
AI-powered media search and editing toolkit.

## Features
- Get clips timestamps from corpus of videos with single query
- Search corpus of images by query
- remove unwanted script lines from the video
- Blur Objects in the video or image
- Add, remove and extract audio from the video
- Cut and compress the video

## Install

install [FFMPEG](https://ffmpeg.org/)
```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

Install python packages:

```bash
pip install git+https://github.com/otman-ai/mediasearch.git@main
```

Check [Whisper Officiel documentations](https://github.com/openai/whisper) if you have any issues installing it.

## How to use

* To search clips in the videos:

```python
from mediasearch.vit import VideoQuery

video_search = VideoQuery()
video_search.insert_videos(videos_path=["path/to/video"]) # insert your videos
results = video_search.search("the query is here") # then you can query 
print("Results:", results)

# Results : [(0, 2) (12.3, 16)]
```

* Search images by query

```python
from mediasearch.vit import ImageQuery

image_query = ImageQuery()
image_query.insert_images(images=["assets/frame.jpg", "assets/frame2.JPG"])
results = image_query.search("buggy")
print(results)
#  {'assets/frame.jpg': 26.54030418395996, 'assets/frame2.JPG': 25.928192138671875}
```

* To remove unwanted words and script from the video 

```python
from mediasearch.edit import CensorText
import whisper

model = whisper.load_model("small")
censor_text = CensorText(model, "video.mp4", "censored_transcript.mp4")

# get the transcript and edit it
transcript = censor_text.get_transcript().replace("No", "") # transcript without the word "No"

censor_text.keep(transcript)

# Output -> video wihout the removed texts in transcript
```

* You can remove, extract audio and cut video

```python
from mediasearch.edit import cut_video, extract_audio, remove_audio, remove_intervals

# cut video
cut_video("assets/video.mp4", "cutted_output.mp4", start_time=0, duration=12)

# extract the audio from the video
extract_audio("video.mp4", "audio.mp3")

# remove the audio from the video
remove_audio("assets/video.mp4", "video_wihout_audio.mp4")

# remove clips from the video
remove_intervals = [(0, 2), (4, 10)]  # the intervals you want to remove
remove_intervals("assets/video.mp4", remove_intervals, video_duration, output_path)
```

* You can blur objects

For a video:
```python
from mediasearch.edit import CensorObjects

censor_objet = CensorObjects(labels=["faces"])
censor_objet.censor_video("video.mp4", "output_blurred.mp4")

# output -> video with blurred faces.
```

For an image
```python
from mediasearch.edit import CensorObjects

censor_objet = CensorObjects(labels=["face"])
censor_objet.censor_image("image.png", "output_blurred_img.png")

# output -> image with blurred faces.
```

### CLI

You can also run the following command to use `mediasearch` from your console:
```bash
# Videos
## Insert video
mediasearch insert-videos "assets/"
## Search video content
mediasearch search-videos "person walking" --threshold 0.05


# Images
## Insert the images
 mediasearch insert-images "assets/"
## Check image similarity
 mediasearch search-images "buggy"

# Censor objects
mediasearch censor video.mp4 censored_transcript.mp4 --labels faces license_plates
```
## License
[License](/LICENSE)