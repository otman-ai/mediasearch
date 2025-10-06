#!/usr/bin/env python3
"""
Setup script for mediasearch package
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

setup(
    name="mediasearch",
    version="0.1.0",
    author="Otman Heddouch",
    author_email="otmanheddouchai@gmail.com",
    description="AI-powered media search and editing toolkit",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/otman-ai/mediasearch",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch==2.8.0",
        "opencv-python==4.12.0.88",
        "numpy==2.2.6",
        "pillow==11.3.0",
        "ffmpeg-python==0.2.0",
        "tqdm==4.67.1",
        "ultralytics==8.3.204",
        "clip @ git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1",
        "openai-whisper @ git+https://github.com/openai/whisper.git@c0d2f624c09dc18e709e37c2ad90c039a4eb72a2",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "mediasearch=mediasearch.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
