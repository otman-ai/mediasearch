#!/usr/bin/env python3
"""
Command-line interface for MediaSearch package.
"""

import argparse
import sys
import os
from typing import Optional

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MediaSearch - AI-powered media search and editing toolkit"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Video search command
    search_parser = subparsers.add_parser("search", help="Search for content in video")
    search_parser.add_argument("video_path", help="Path to video file")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--threshold", type=float, default=0.02, help="Similarity threshold")
    search_parser.add_argument("--model", default="ViT-B/32", help="CLIP model to use")
    
    # Image similarity command
    image_parser = subparsers.add_parser("image-similarity", help="Check image similarity to query")
    image_parser.add_argument("image_path", help="Path to image file")
    image_parser.add_argument("query", help="Search query")
    image_parser.add_argument("--model", default="ViT-B/32", help="CLIP model to use")
    
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
        if args.command == "search":
            from .vit import VideoText
            video_search = VideoText(
                video_path=args.video_path,
                model_name=args.model,
                threshold=args.threshold
            )
            results = video_search.search(args.query)
            print(f"Found {len(results)} matching segments:")
            for start, end in results:
                print(f"  {start:.2f}s - {end:.2f}s")
                
        elif args.command == "image-similarity":
            from .vit import TextImage
            image_sim = TextImage(model_name=args.model)
            result = image_sim(args.image_path, args.query)
            print(f"Similarity score: {result['probs']:.4f}")
            
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
