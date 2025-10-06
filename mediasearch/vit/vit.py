from PIL import Image
import torch
import clip
import numpy as np
import cv2
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

__all__ = [
    "VideoText",
    "TextImage",
    "MODELS"
]

MODELS = [
    "RN50",
    "RN101",
    "RN50x4",
    "RN50x16",
    "RN50x64",
    "ViT-B/32",
    "ViT-B/16",
    "ViT-L/14",
    "ViT-L/14@336px"
]
class VideoText:
    """Highlight the parts of the video that matches with the query"""
    def __init__(self, video_path:str="assets/video.mp4", model_name:str="ViT-B/32", frame_rate:int=30,  threshold:float=0.02):
        self.model_name = model_name
        self.threshold = threshold
        self.frame_rate = frame_rate
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.video_path = video_path
        if not os.path.isfile(self.video_path):
            raise FileNotFoundError(f"The video path {self.video_path} deos not exists")
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.logits = []
        self.probs = None
        self.rate_seoncd = self.frame_rate / self.fps

    def __call__(self, *args, **kwds):
        return self.search( *args, **kwds)
    
    def search(self, query: str) -> list:
        timestamps_extracted = []
        logging.info("Tokenizing the query...")
        # tokenize the query
        query_tokenized = clip.tokenize([query]).to(self.device)
        frame_count = 0
        logging.info(f"Start reading the video from {self.video_path}")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_count % self.frame_rate == 0:
                img = Image.fromarray(frame).convert("RGB")
                img_feature = self.preprocess(img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logit, _ = self.model(img_feature, query_tokenized)
                    self.logits.append(float(logit.cpu().numpy()[0]))
                
            frame_count += 1

        self.probs = torch.tensor(self.logits).softmax(dim=0).numpy()
        self.cap.release()
        logging.info(f"{self.logits}")
        rangs = np.array(list(range(0, len(self.logits))))
        matched_frame_indics = rangs[self.probs > self.threshold]

        if len(matched_frame_indics) > 0:
            logging.info("found Matched frame")
            logging.info(matched_frame_indics)
            timestamps_extracted = [
                (key * self.rate_seoncd,
                 key * self.rate_seoncd + self.rate_seoncd)
                for key in matched_frame_indics 
            ]
            logging.info("Time stamps extraced with success.")

        return timestamps_extracted
    

class TextImage:
    """How an image is related to a query"""

    def __init__(self,  model_name:str="ViT-B/32"):
        self.model_name = model_name
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)

    def __call__(self, image_path:str, query:str="black guy"):
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"The image {image_path} deos not exists")
            logging.info("Tokenizing the query...")
            logits = None
            # tokenize the query
            query_tokenized = clip.tokenize([query]).to(self.device)
            img = Image.open(image_path)
            logging.info("Preprocessing the image...")
            img_feature = self.preprocess(img).unsqueeze(0).to(self.device)
            logging.info("Making the prediction..")
            with torch.no_grad():
                logits, _ = self.model(img_feature, query_tokenized)
            logging.info("Finished processing.")
            return {"probs":logits.squeeze(0).cpu().numpy()[0]}