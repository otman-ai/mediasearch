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

logger = logging.getLogger("mediasearch.vit")

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
                 frame_rate:int=10,
                 threshold:float=0.25,
                 cash=video_embeddings_path,
                 debug:bool=False):
        self.model_name = model_name
        self.threshold = threshold
        self.frame_rate = frame_rate
        self.debug = debug
        self.logger = logger
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.logits = []
        self.probs = None
        self.cash = cash
        self.video_embeddings= {}
        os.makedirs(cash_dir, exist_ok=True)

    def __call__(self, *args, **kwds):
        return self.search( *args, **kwds)

    def insert_videos(self, videos_path:List=[]):
        logging.info(f"Inserting {videos_path} videos")
        with h5py.File(self.cash, "a") as f:
            groups = [key for key in f.keys() if isinstance(f[key], h5py.Group)]
            for idx, video in tqdm(enumerate(videos_path,start=len(groups))):
                if not os.path.isfile(video):
                    raise FileNotFoundError
                cap = cv2.VideoCapture(video)
                fps = cap.get(cv2.CAP_PROP_FPS)
                videoDuration =float(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) /fps)
                self.logger.debug("Video duration is %s", videoDuration)
                grp = f.create_group(str(idx))
                grp.create_dataset("video", data=[video])
                grp.create_dataset("fps", data=[fps])
                grp.create_dataset("rate_second", data=[self.frame_rate / fps])
                grp.create_dataset("duration", data=[videoDuration])
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
                        img_feature = self.preprocess(img).unsqueeze(0)
                        self.logger.debug("Image feature shape before encoding: %s", img_feature.shape)
                        with torch.no_grad():
                            img_feature = self.model.encode_image(img_feature)
                            self.logger.debug("Image feature shape after encoding: %s", img_feature.shape)
                            img_feature /= img_feature.norm(dim=-1, keepdim=True)
                            self.logger.debug("Image feature shape after normalization: %s", img_feature.shape)
                        # save the img features as npy
                        #self.video_embeddings[idx]["embeddings"][frame_index] = img_feature.tolist()
                        video_embeddings.append(img_feature.tolist())
                        frame_index += 1
                    frame_count += 1
                logging.info(f"Finished reading the video from {video}")
                cap.release()
                self.logger.debug("Video embeddings shape: %s", np.array(video_embeddings).shape)
                grp.create_dataset("embeddings", data=np.array(video_embeddings, dtype=np.float32))
            #with open(self.cash, "w") as f:
            #    json.dump(self.video_embeddings, f)
            logging.info(f"The embeddings saved to {self.cash}")

    def search(self, query: str) -> dict | None:
        timestamps_extracted = {}
        logging.info("Tokenizing the query...")
        # tokenize the query
        query_tokenized = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            encoded_query  = self.model.encode_text(query_tokenized)
            encoded_query /= encoded_query.norm(dim=-1, keepdim=True)
        self.logger.debug("Encoded query shape: %s", encoded_query.shape)
        if not os.path.exists(self.cash) and os.path.getsize(self.cash) == 0:
            self.logger.warning("The embeddings file is empty. Please insert videos first.")
            return
        all_hits = []
        with h5py.File(self.cash, "r") as f:
            for key in f.keys():
                self.logger.debug("Key: %s", f[key].keys())
                # get all the embeddings of the video
                embeddings = np.array(f[key]["embeddings"][:]).squeeze(1)
                self.logger.debug("Embedding shape: %s", embeddings.shape)
                rate_second = f[key]["rate_second"][:][0]
                # loop through each embedded frame
                sims = (encoded_query.detach().cpu().numpy() @ embeddings.T).ravel()
                self.logger.debug("Sims %s", sims)
                video = f[key]["video"][0].decode('utf-8')
                duration = f[key]["duration"][:][0]
                for i, s in enumerate(sims):
                    all_hits.append((video, i, float(s), float(rate_second), duration))

        max_sim = max(h[2] for h in all_hits)
        kept = [h for h in all_hits
                if h[2] >= self.threshold and h[2] >= max_sim - 0.03]
        for k in kept:
            timestamps_extracted[k[0]] = timestamps_extracted.get(k[0], []) + [(float(k[1] * k[3]), min(float(k[1] * k[3] + k[3]), float(k[4])))]

        return timestamps_extracted
    

class ImageQuery:
    """How an image is related to a query"""

    def __init__(self,  model_name:str="ViT-B/32", cash=image_embeddings_path, debug:bool=False, threshold:float=0.25, logger=logging.getLogger("mediasearch.vit")):
        self.model_name = model_name
        self.cash = cash
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.debug = debug
        self.logger = logger
        self.threshold = threshold
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        
    def insert_images(self, images:List=None):
        self.logger.info(f"Inserting {len(images)} images")
        with h5py.File(self.cash, "a") as f:
            groups = [key for key in f.keys() if isinstance(f[key], h5py.Group)]
            self.logger.debug("Images: %s", images)
            for idx, image in tqdm(enumerate(images,start=len(groups))):
                self.logger.info(f"Preprocessing the image {image} with index {idx}...")
                if not os.path.isfile(image):
                    raise FileNotFoundError
                self.logger.info("Opening...")
                img = Image.open(image).convert("RGB")
                img_features = self.preprocess(img).unsqueeze(0)
                with torch.no_grad():
                    img_features = self.model.encode_image(img_features)
                    self.logger.debug("Image feature shape after encoding: %s", img_features.shape)
                    img_features /= img_features.norm(dim=-1, keepdim=True)
                    self.logger.debug("Image feature shape after normalization: %s", img_features.shape)
                self.logger.debug(f"Embeddings: {img_features.shape}")
                grp = f.create_group(str(idx))
                grp.create_dataset("image", data=[image])
                grp.create_dataset("embeddings", data=np.asarray(img_features, dtype=np.float32))
                self.logger.debug(f"The embeddings for image {image} saved")
        self.logger.info(f"The embeddings saved to {self.cash}")


    def search(self, query:str):
            if not os.path.isfile(self.cash):
                raise FileNotFoundError("Cash does not exist")
            self.logger.info("Tokenizing the query...")
            logits = {}
            # tokenize the query
            query_tokenized = clip.tokenize([query]).to(self.device)
            with torch.no_grad():
                encoded_query  = self.model.encode_text(query_tokenized)
                encoded_query /= encoded_query.norm(dim=-1, keepdim=True)
            with h5py.File(self.cash, "r") as f:
                for key in f.keys():
                    img = f[key]["image"][0].decode('utf-8')
                    self.logger.info(f"Search the image: {img}")
                    embeddings = np.array(f[key]["embeddings"][:])
                    self.logger.debug("Embedding shape: %s", embeddings.shape)
                    self.logger.debug("Encoded query shape: %s", encoded_query.shape)
                    sims = (encoded_query.detach().cpu().numpy() @ embeddings.T).ravel()[0]
                    self.logger.debug("Sims %s", sims)
                    if sims >= self.threshold:
                        logits[img] = float(sims)
                self.logger.info("Finished processing.")
                return logits

