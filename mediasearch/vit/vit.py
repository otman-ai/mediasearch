import json
from typing import List

from PIL import Image
import torch
import clip
import numpy as np
import cv2
import logging
import h5py
import os
from tqdm import tqdm
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

__all__ = [
    "VideoQuery",
    "ImageQuery",
    "MODELS",
    "video_embeddings_path",
    "image_embeddings_path"
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
default = os.path.join(os.path.expanduser("~"), ".cache")
cash_dir = os.path.join(os.getenv("XDG_CACHE_HOME", default), "mediasearch")
video_embeddings_path = os.path.join(cash_dir, "embeddings.h5")
image_embeddings_path =  os.path.join(cash_dir, "image_embeddings.h5")
class VideoQuery:
    """Highlight the parts of the video that matches with the query"""
    def __init__(self,
                 model_name:str="ViT-B/32",
                 frame_rate:int=30,
                 threshold:float=0.02,
                 cash=video_embeddings_path):
        self.model_name = model_name
        self.threshold = threshold
        self.frame_rate = frame_rate
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.logits = []
        self.probs = None
        self.cash = cash
        self.video_embeddings= {}

    def __call__(self, *args, **kwds):
        return self.search( *args, **kwds)

    def insert_videos(self, videos_path:List=None):
        logging.info(f"Inserting {videos_path} videos")
        with h5py.File(self.cash, "a") as f:
            groups = [key for key in f.keys() if isinstance(f[key], h5py.Group)]
            for idx, video in tqdm(enumerate(videos_path,start=len(groups))):
                if not os.path.isfile(video):
                    raise FileNotFoundError
                cap = cv2.VideoCapture(video)
                fps = cap.get(cv2.CAP_PROP_FPS)
                grp = f.create_group(str(idx))
                grp.create_dataset("video", data=[video])
                grp.create_dataset("fps", data=[fps])
                grp.create_dataset("rate_second", data=[self.frame_rate / fps])
                frame_count = 0
                frame_index = 0
                logging.info(f"Start reading the video from {video}")
                video_embeddings = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % self.frame_rate == 0:
                        img = Image.fromarray(frame).convert("RGB")
                        img_feature = self.preprocess(img).unsqueeze(0).to(self.device).numpy()
                        # save the img features as npy
                        #self.video_embeddings[idx]["embeddings"][frame_index] = img_feature.tolist()
                        video_embeddings.append(img_feature.tolist())
                        frame_index += 1
                    frame_count += 1
                logging.info(f"Finished reading the video from {video}")
                cap.release()
                grp.create_dataset("embeddings", data=np.array(video_embeddings, dtype=np.float32))
            #with open(self.cash, "w") as f:
            #    json.dump(self.video_embeddings, f)
            logging.info(f"The embeddings saved to {self.cash}")

    def search(self, query: str) -> dict:
        timestamps_extracted = {}
        logging.info("Tokenizing the query...")
        # tokenize the query
        query_tokenized = clip.tokenize([query]).to(self.device)
        with h5py.File(self.cash, "r") as f:
            for key in f.keys():
                video_logits = []
                # get all the embeddings of the video
                embeddings = np.array(f[key]["embeddings"][:])
                rate_second = f[key]["rate_second"][:][0]
                # loop through each embedded frame
                for embedding_idx in range(0, embeddings.shape[0]):
                    with torch.no_grad():
                        logit, _ = self.model(torch.tensor(embeddings[embedding_idx]).to(self.device),
                                              query_tokenized)
                        print(logit.cpu().numpy()[0][0])
                        video_logits.append(float(logit.cpu().numpy()[0][0]))
                probs = torch.tensor(np.array(video_logits)).softmax(dim=0).numpy()
                ranges = np.array(list(range(0, len(video_logits))))
                matched_frame_indices = ranges[probs > self.threshold]
                video = f[key]["video"][0].decode('utf-8')
                if len(matched_frame_indices) > 0:
                    logging.info(video)
                    timestamps_extracted[video] = [
                        (float(k * rate_second),
                         float(k * rate_second + rate_second))
                        for k in matched_frame_indices
                    ]
                    logging.info(f"Time stamps extracted with success for video {video}.")

        return timestamps_extracted
    

class ImageQuery:
    """How an image is related to a query"""

    def __init__(self,  model_name:str="ViT-B/32", cash=image_embeddings_path):
        self.model_name = model_name
        self.cash = cash
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)

    def insert_images(self, images:List=None):
        logging.info(f"Inserting {len(images)} images")
        with h5py.File(self.cash, "a") as f:
            groups = [key for key in f.keys() if isinstance(f[key], h5py.Group)]
            print("Images:", images)
            for idx, image in tqdm(enumerate(images,start=len(groups))):
                logging.info(f"Preprocessing the image {image} with index {idx}...")
                if not os.path.isfile(image):
                    raise FileNotFoundError
                logging.info("Opening...")
                img = Image.open(image)
                img_features = self.preprocess(img).unsqueeze(0).cpu().numpy()
                logging.info(f"Embeddings: {img_features.shape}")
                grp = f.create_group(str(idx))
                grp.create_dataset("image", data=[image])
                grp.create_dataset("embeddings", data=np.array(img_features, dtype=np.float32))
                logging.info(f"The embeddings for image {image} saved")
        logging.info(f"The embeddings saved to {self.cash}")


    def search(self, query:str):
            if not os.path.isfile(self.cash):
                raise FileNotFoundError("Cash does not exist")
            logging.info("Tokenizing the query...")
            logits = {}
            # tokenize the query
            query_tokenized = clip.tokenize([query]).to(self.device)

            with h5py.File(self.cash, "r") as f:
                for key in f.keys():
                    img = f[key]["image"][0].decode('utf-8')
                    logging.info(f"Search the image: {img}")
                    img_features = torch.tensor(f[key]["embeddings"][:]).to(self.device)
                    with torch.no_grad():
                        logit, _ = self.model(img_features, query_tokenized)
                        logits[img] = float(logit.squeeze(0).cpu().numpy()[0])
                logging.info("Finished processing.")
                return logits

