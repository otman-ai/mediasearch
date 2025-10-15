#!/usr/bin/env python3
"""
Command-line interface for MediaSearch package.
"""

import argparse
import logging
import sys
import os
from mediasearch.vit import video_embeddings_path, image_embeddings_path
from typing import Optional

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MediaSearch - AI-powered media search and editing toolkit"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Video insert command
    insert_parser = subparsers.add_parser("insert-videos", help="Search for content in corpus of videos")
    insert_parser.add_argument("input_path", help="Paths to videos")
    insert_parser.add_argument("--cash", help="Paths to cash the embeddings", default=video_embeddings_path)
    insert_parser.add_argument("--threshold", type=float, default=0.02, help="Similarity threshold")
    insert_parser.add_argument("--model", default="ViT-B/32", help="CLIP model to use")

    # Video search command
    search_parser = subparsers.add_parser("search-videos", help="Search for content in videos")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--cash", help="Paths where the embeddings", default=video_embeddings_path)
    search_parser.add_argument("--threshold", type=float, default=0.02, help="Similarity threshold")
    search_parser.add_argument("--model", default="ViT-B/32", help="CLIP model to use")

    # Insert images command
    image_parser = subparsers.add_parser("insert-images", help="Insert images to database for retrieval")
    image_parser.add_argument("input_path", help="Path to image files")
    image_parser.add_argument("--model", default="ViT-B/32", help="CLIP model to use")
    image_parser.add_argument("--cash", help="Where to save the embeddings", default=image_embeddings_path, )

    # Image similarity command
    image_parser = subparsers.add_parser("search-images", help="Check how similar all the image to query")
    image_parser.add_argument("query", help="Search query")
    image_parser.add_argument("--model", default="ViT-B/32", help="CLIP model to use")
    image_parser.add_argument("--cash", help="Where the embeddings saved", default=image_embeddings_path, )

    # Censor objects command
    censor_parser = subparsers.add_parser("censor", help="Censor objects in video/image")
    censor_parser.add_argument("input_path", help="Path to input file")
    censor_parser.add_argument("output_path", help="Path to output file")
    censor_parser.add_argument("--labels", nargs="+", default=["faces"], help="Objects to censor")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "search-videos":
            from .vit import VideoQuery
            video_search = VideoQuery(
                model_name=args.model,
                threshold=args.threshold,
                cash=args.cash,
            )
            results = video_search.search(args.query)
            print(f"Found {len(results)} matching segments:")
            print("Results:", results)

        elif args.command =="insert-videos":
            from .vit import VideoQuery
            import glob
            videos = []
            video_patterns = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.mpg", "*.m4v"]
            for pattern in video_patterns:
                videos.extend(glob.glob(os.path.join(args.input_path, pattern)))
            logging.info("Found {} videos {}".format(len(videos), videos))
            video_search = VideoQuery(
                model_name=args.model,
                threshold=args.threshold,
                cash=args.cash,
            )
            results = video_search.insert_videos(videos_path=videos)
            print(results)

        elif args.command == "insert-images":
            from .vit import ImageQuery
            import glob
            images = []
            image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp","*.gif"]
            for pattern in image_patterns:
                images.extend(glob.glob(os.path.join(args.input_path, pattern)))
            logging.info("Found {} images {}".format(len(images), images))
            image_search = ImageQuery(
                model_name=args.model,
                cash=args.cash)
            results = image_search.insert_images(images=images)
            print("Done")
        elif args.command == "search-images":
            from .vit import ImageQuery
            image_search = ImageQuery(model_name=args.model,
                                   cash=args.cash)
            result = image_search.search(args.query)
            print("Results:",result)
            
        elif args.command == "censor":
            from .edit import CensorObjects
            censor_obj = CensorObjects(labels=args.labels)
            
            # Check if input is video or image
            ext = os.path.splitext(args.input_path)[1].lower()
            if ext in ['.mp4', '.mkv', '.mov', '.avi', '.webm']:
                censor_obj.censor_video(args.input_path, args.output_path)
            elif ext in ['.png', '.jpg', '.jpeg']:
                censor_obj.censor_image(args.input_path, args.output_path)
            else:
                print(f"Unsupported file format: {ext}")
                sys.exit(1)
                
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
