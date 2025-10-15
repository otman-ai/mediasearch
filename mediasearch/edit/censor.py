import os
import re
from whisper import Whisper
from .helpers import (add_audio, get_video_duration, 
                      remove_intervals, blur_region,
                      compression)
import cv2

from ultralytics import YOLO
import urllib
from tqdm import tqdm

default = os.path.join(os.path.expanduser("~"), ".cache")
download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "mediasearch")



_VS_LABELS_ = {
    "faces" :{
        "name":  "face-detection.pt",
        "url":"https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt"
    },
    "license_plates" :{
        "name":  "license-plate.pt",
        "url":"https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1n.pt"
    }
}

SUPPORTED_VIDEO_EXTENSIONS = [
    "mp4", 
    "mkv", 
    "mov", 
    "avi",
    "webm"
]

SUPPORTED_IMAGE_EXTENSIONS = [
    "png",
    "jpg",
    "jpeg"
]


def _download(url: str, root: str, name:str, in_memory: bool=False) -> str:
    os.makedirs(root, exist_ok=True)

    download_target = os.path.join(root, name)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return download_target

class CensorText:
    def __init__(self, model:Whisper, source:str, output_file:str):
        self.model = model
        self.transcript_words = []
        self.subtitles_word = {}
        self.time_range_to_cut = []
        self.merged  = []
        self.segments = []
        self.source = source
        if not os.path.isfile(self.source):
            raise FileNotFoundError(f"The file {self.source} is not exists")
        self.video_duration = get_video_duration(self.source)
        self.output_file = output_file
        self.edited_script_list_word = []
        self.segments = self.model.transcribe(self.source, word_timestamps=True)
        self.transcript = self.segments["text"]

    def get_transcript(self):
        return self.transcript
    
    def keep(self, transcript:str):
    
        edited_transcript = re.sub(r'[^\w\s]', '', transcript)
        self.edited_script_list_word = [i for i in edited_transcript.split(' ') if i != '']
        self.mapping_segments()
        self.find_time_range_cutted()
        self.merge_intervals()
        remove_intervals(self.source, self.merged, self.video_duration, self.output_file)

    def mapping_segments(self):
        """
        Mapped the subtitles, each word with it correspond start and end time
        """
        # Empty dictionary to store the subtitles with it's own start and end time
        # looping for every segments
        for segment in self.segments["segments"]:
            for word in segment["words"]:
                # clean the word from any space or punctuation.
                text_without_punctuation = re.sub(r'[^\w\s]', '', word["word"].strip())
                # Store the cleaned word in dic
                self.subtitles_word[f"{word["start"]}-{word["end"]}"] =  text_without_punctuation
                # as well as in list
                self.transcript_words.append(text_without_punctuation)


    def find_time_range_cutted(self):
        # assign 0 to tracked_index which track the word index in original script with index in new edited script
        tracked_index = 0
        # empty list to store the time range to cut
        # loop through all the original word
        for range_, sub in self.subtitles_word.items():
            # get the correspond word of the new script
            compared_value = self.edited_script_list_word[tracked_index]
            # if the index of old script is  equal to new script then it hasnt cutted move to the next word.
            if sub == compared_value:
                tracked_index += 1
            # otherwise add its range as it removed by th user and assign  tracked_index same number as it is
            # This is will not shift the index of the new script until we found its own range from the old one
            else :
                self.time_range_to_cut.append(range_)
        self.time_range_to_cut = [(float(i.split('-')[0]), float(i.split('-')[1])) for i in self.time_range_to_cut]
   
    def merge_intervals(self):
        if not self.time_range_to_cut:
            return []
        
        # Sort by start time
        self.time_range_to_cut.sort(key=lambda x: x[0])
        self.merged = [self.time_range_to_cut[0]]
        
        for start, end in self.time_range_to_cut[1:]:
            last_start, last_end = self.merged[-1]
            
            # If intervals overlap or touch, merge them
            if start <= last_end:
                self.merged[-1] = (last_start, max(last_end, end))
            else:
                self.merged.append((start, end))

class CensorObjects:
    """Censor video and blure objects"""
    def __init__(self, labels:list=['faces'], show=False):
        self.labels = labels
        self.show = show
        for label in self.labels:
            if label not in _VS_LABELS_.keys():
                raise KeyError(f"The label '{label}' is unkown and not in {list(_VS_LABELS_.keys())}")
        self.models = {
            label: YOLO(_download(_VS_LABELS_[label]["url"], download_root, _VS_LABELS_[label]["name"]))
            for label in labels
        }
    
    def censor_video(self, source:str, output_file:str):
        output_without_audio = "output_without_audio.mp4"
        output_without_compression = "compression.mp4"
        if not os.path.isfile(source):
            raise FileNotFoundError(f"The source file {source} deos not exists")
        
        if source.split(".")[-1] not in SUPPORTED_VIDEO_EXTENSIONS:
            raise NotImplementedError(f"The extension {source.split('.')[-1]} is not supported {SUPPORTED_VIDEO_EXTENSIONS}")
        
        cap = cv2.VideoCapture(source)
        # Get video properties
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec for mp4
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define output video writer
        out = cv2.VideoWriter(output_without_audio, fourcc, fps, (width, height))
        if not cap.isOpened():
            raise Exception(f"Could not open the source file {self.source}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            for _, m in self.models.items():
                results = m(frame, verbose=False)
                for r in results:
                    if getattr(r, "boxes"):
                        for box in r.boxes:
                            x, y, w, h = map(int, box.xyxy[0])
                            frame = blur_region(frame, x, y, w, h)
                
            out.write(frame)
            
            if self.show:
                cv2.imshow("Video", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        # Release everything
        cap.release()
        out.release()
        if self.show:
            cv2.destroyAllWindows()
        add_audio(output_without_audio, source, output_without_compression)
        compression(output_without_compression, output_file)

        os.remove(output_without_audio)
        os.remove(output_without_compression)

        print(f"Video saved to {output_file}")

    def censor_image(self, source:str, output_file:str):
        if not os.path.isfile(source):
            raise FileNotFoundError(f"The source file {source} deos not exists")
        
        if source.split(".")[-1] not in SUPPORTED_IMAGE_EXTENSIONS:
            raise NotImplementedError(f"""The extension {source.split('.')[-1]} is not 
                                          supported {SUPPORTED_VIDEO_EXTENSIONS}""")
        
        # Convert to grayscale for detection
        img = cv2.imread(source)
        if img is None:
            raise Exception("The image is None and not supported")
        for l, m in self.models.items():
            results = m(img, verbose=False)
            for r in results:
                for box in r.boxes:
                    x, y, w, h = map(int, box.xyxy[0])
                    img = blur_region(img, x, y, w, h)
        
        # Write processed frame to output
        cv2.imwrite(output_file, img)
        print(f"Image saved to {output_file}")
