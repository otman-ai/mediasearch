# Media Search
AI-powered media search and editing toolkit.

## Features
- Get sub-clips that coorespond to your query without transcribing the entire video
- Check how similar an image with query
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
pip install git+http://github.com/otman-ai/mediasearch.git
```
Check [Whisper Officiel documentations](https://github.com/openai/whisper) if you have any issues installing it.

## How to use

* To get parts of the video that coorespond to your query 

```python
from mediasearch.vit import VideoText

video_search = VideoText(video_path="path/to/video")
results = video_search.search("the query is here")
print("Results:", results)

# Results : [(0, 2) (12.3, 16)]
```

* To see how an image is correlated to your query

```python
from mediasearch.vit import TextImage

image_sm = TextImage()

results = image_sm("path/to/image", "your query here")
print(res)

# {'probs':13.98}
```

* To remove unwanted words and script from the video 

```python
from mediasearch.edit import CensorText
import whisper

model = whisper.load_model("small")
censor_text = CensorText(model, "video.mp4", "output.mp4")

# get the transcript and edit it
transcript = censor_text.get_transcript().replace("No", "") # transcript without the word "No"

censor_text.keep(transcript)

# Output -> video wihout the removed texts in transcript
```

* You can remove, extract audio and cut video 

```python
from mediasearch.edit import cut_video, extract_audio, remove_audio, remove_intervals

# cut video
cut_video("video.mp4", "cutted_output.mp4", start_time=0, duration=12)

# extract the audio from the video
extract_audio("video.mp4", "audio.mp3")

# remove the audio from the video
remove_audio("video.mp4", "video_wihout_audio.mp4")

# remove clips from the video
remove_intervals = [(0, 2), (4, 10)] # the intervals you want to remove
remove_intervals(video_path, remove_intervals, video_duration, output_path)
```

* You can blur objects

For a video:
```python
from mediasearch.edit import CensorObjectS

censor_objet = CensorObjectS(labels=["faces"])
censor_objet.censor_video("video.mp4", "output_blurred.mp4")

# output -> video with blurred faces.
```

For an image
```python
from mediasearch.edit import CensorObjectS

censor_objet = CensorObjectS(labels=["face"])
censor_objet.censor_image("image.png", "output_blurred_img.png")

# output -> image with blurred faces.
```

### CLI

You can also run the following command to use `mediasearch` from your console:
```bash
# Search video content
mediasearch search video.mp4 "person walking" --threshold 0.05

# Check image similarity
mediasearch image-similarity image.jpg "cat sitting"

# Censor objects
mediasearch censor video.mp4 output.mp4 --labels faces license_plates
```
## License
[License](/LICENSE)