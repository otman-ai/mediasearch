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
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

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
                 batch_size: int = 32,
                 num_workers: int | None = None,
                 debug:bool=False):
        
        self.batch_size = batch_size
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"   
        if self.device == "cuda":
            self.num_workers = num_workers or min(8, os.cpu_count() or 4)
        else:
            self.num_workers = num_workers or 2
        cv2.setNumThreads(self.num_workers)
        torch.set_num_threads(os.cpu_count())
        self.model_name = model_name
        self.threshold = threshold
        self.frame_rate = frame_rate
        self.debug = debug
        self.logger = logger
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.logits = []
        self.probs = None
        self.cash = cash
        self.video_embeddings= {}
        os.makedirs(cash_dir, exist_ok=True)

    def __call__(self, *args, **kwds):
        return self.search( *args, **kwds)

    def _preprocess_frame(self, frame):
        img = Image.fromarray(frame).convert("RGB")
        return self.preprocess(img)
    
    def _encode_batch(self, frames, pool):
        tensors = list(pool.map(self._preprocess_frame, frames))
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            img_features = self.model.encode_image(batch)
            img_features /= img_features.norm(dim=-1, keepdim=True)
        return img_features.cpu()
    
    def _embed_video(self, video:str) -> torch.Tensor:

        cap = cv2.VideoCapture(video)
        embeddings, batch, fc = [], [], 0

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if fc % self.frame_rate == 0:
                    batch.append(frame)
                fc += 1
                if len(batch) == self.batch_size:
                    img_features = self._encode_batch(batch, pool)
                    embeddings.append(img_features)
                    batch = []
            if batch:
                img_features = self._encode_batch(batch, pool)
                embeddings.append(img_features)
        cap.release()
        return torch.cat(embeddings) if embeddings else torch.empty(0)
    
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
                cap.release()

                self.logger.debug("Video duration is %s", videoDuration)
                grp = f.create_group(str(idx))
                grp.create_dataset("video", data=[video])
                grp.create_dataset("fps", data=[fps])
                grp.create_dataset("rate_second", data=[self.frame_rate / fps])
                grp.create_dataset("duration", data=[videoDuration])
                logging.info(f"Start reading the video from {video}")
                # pipeline
                video_embeddings = self._embed_video(video)

                logging.info(f"Finished reading the video from {video}")
                cap.release()
                grp.create_dataset("embeddings", data=np.asarray(video_embeddings.cpu(), dtype=np.float32), compression="lzf")

            logging.info(f"The embeddings saved to {self.cash}")

    def search(self, query: str, is_united_timestamp: bool=True) -> dict | None:
        logging.info("Tokenizing the query...")

        if not os.path.exists(self.cash) and os.path.getsize(self.cash) == 0:
            self.logger.warning("The embeddings file is empty. Please insert videos first.")
            return
        with h5py.File(self.cash, "r") as f:
            keys = list(f.keys())
            embds, meta = [], []
            if not keys:
                self.logger.warning("The embeddings file is empty. Please insert videos first.")
                return
            for key in keys:
                # get all the embeddings of the video
                e = f[key]["embeddings"][:]
                rate_second = f[key]["rate_second"][:][0]
                # loop through each embedded frame
                video = f[key]["video"][0].decode('utf-8')
                duration = f[key]["duration"][:][0]
                meta.append((video, rate_second, duration, len(e)))
                embds.append(e)

                # for i, s in enumerate(sims):
                #     all_hits.append((video, i, float(s), float(rate_second), duration))
        all_embds = torch.from_numpy(np.concatenate(embds, axis=0)).to(self.device)
        # tokenize the query
        query_tokenized = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            encoded_query  = self.model.encode_text(query_tokenized)
            encoded_query /= encoded_query.norm(dim=-1, keepdim=True)

        self.logger.debug("Encoded query shape: %s", all_embds.dtype)
        with torch.autocast(device_type=self.device, dtype=torch.float16):
            sims = (all_embds @ encoded_query.T).cpu().numpy().ravel()
        self.logger.debug("Sims shape: %s", sims.shape)
        max_sim = sims.max()
        keep = (sims >=  self.threshold) & (sims >= max_sim - 0.03)
        out, offset = {}, 0
        for video, rate, duration, n in meta:
            idx = np.nonzero(keep[offset:offset + n])[0]   # only kept frames
            if idx.size:
                starts = idx * rate
                ends = np.minimum(starts + rate, duration)
                scores = sims[offset:offset + n][idx]
                out.setdefault(video, []).extend(
                    (float(a), float(b), float(c))
                    for a, b, c in zip(starts, ends, scores))
            offset += n
        if not is_united_timestamp:
            return out
        united_timestamps = {}
        for key, value in out.items():
            united_timestamps[key] = []
            for idx, v in enumerate(value):
                start, end, score = v
                if int(start) == int(value[idx-1][1]):
                    print("Video", video)
                    united_timestamps[key][-1] = (united_timestamps[key][-1][0], end, score)
                else:
                    united_timestamps[key].append((start, end, score))
                
        return united_timestamps
    

class ImageQuery:
    """How an image is related to a query"""

    def __init__(self,  model_name:str="ViT-B/32", batch_size:int=32, cash=image_embeddings_path, debug:bool=False, threshold:float=0.25, logger=logging.getLogger("mediasearch.vit")):
        self.model_name = model_name
        self.batch_size = batch_size
        self.cash = cash
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.debug = debug
        self.logger = logger
        self.threshold = threshold
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        
    def _preprocess_frame(self, image):
        img = Image.open(image).convert("RGB")
        return self.preprocess(img)
    
    def _encode_image(self, img_features) -> torch.Tensor:
        with torch.no_grad():
            img_features = self.model.encode_image(img_features.to(self.device))
            self.logger.debug("Image feature shape after encoding: %s", img_features.shape)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            self.logger.debug("Image feature shape after normalization: %s", img_features.shape)
        return img_features.cpu()

    def _encode_batches(self, images, start):
        batch = []
        embeddings_batches = []
        for idx, image in tqdm(enumerate(images,start=start)):
            self.logger.info(f"Preprocessing the image {image} with index {idx}...")
            if not os.path.isfile(image):
                raise FileNotFoundError
            
            self.logger.info("Opening...")
            img_features = self._preprocess_frame(image)
            batch.append(img_features)
            
            if len(batch) == self.batch_size:
                embeddings_batches.append(self._encode_image(torch.stack(batch)))
                batch = []
        if batch:
            embeddings_batches.append(self._encode_image(torch.stack(batch)))
        return torch.cat(embeddings_batches) if embeddings_batches else torch.empty(0)


    def insert_images(self, images:List=None):
        self.logger.info(f"Inserting {len(images)} images")
        with h5py.File(self.cash, "a") as f:
            groups = [key for key in f.keys() if isinstance(f[key], h5py.Group)]
            self.logger.debug("Images: %s", images)
            start = len(groups)
            embeddings_batches = self._encode_batches(images, start=start)
            self.logger.debug(f"Embeddings: {embeddings_batches.shape}")
            idx_batch = 0
            for idx, image in tqdm(enumerate(images,start=start)):
                grp = f.create_group(str(idx))
                grp.create_dataset("image", data=[image])
                grp.create_dataset("embeddings", data=np.asarray(embeddings_batches[idx_batch], dtype=np.float32), compression="lzf")
                self.logger.debug(f"The embeddings for image {image} saved")
                idx_batch += 1
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
                    embeddings = f[key]["embeddings"][:]
                    self.logger.debug("Embedding shape: %s", embeddings.shape)
                    self.logger.debug("Encoded query shape: %s", encoded_query.shape)
                    sims = (encoded_query.detach().cpu().numpy() @ embeddings.T).ravel()[0]
                    self.logger.debug("Sims %s", sims)
                    if sims >= self.threshold:
                        logits[img] = float(sims)
                self.logger.info("Finished processing.")
                return logits

