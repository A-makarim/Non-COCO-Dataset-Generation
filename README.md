# Mars Probe Detection

An end-to-end computer vision pipeline to **automatically generate a YOLOv8 dataset**
for detecting **ERC Mars Probes** from **unlabelled videos or images**, using  
**SAM3 (Segment Anything Model 3 by Meta)** for auto-annotation of non-COCO or unsual classes.

I used this project for ERC Mars Probes. This project work for any **non-COCO object classes**, where labelled datasets do not exist. The entire outline automates the process. Just run it on a virtual machine and grab a drink while YOLO learns from SAM3.

## Motivation

I had to run YOLOV8n on NVIDIA Jetson to find a good balance between accuracy and computaiton time. Using SAM3 is not optimal on such devices due to number of params. YOLO itself didn't had the intelligence to detect a prompt Object. It is not a ViT like SAM3. 

ERC Mars Probes are not part of standard datasets such as COCO.  
Manually annotating thousands of frames is slow, error-prone, and expensive.

This repository solves that problem by:
- Using **SAM3** to segment objects via text prompts
- Converting SAM3 outputs into **YOLO-format bounding boxes**
- Automatically building a **train/val/test dataset**
- Training a YOLOv8 detector on the generated data

[to be added]
- model hallucination/edge cases for poorly defined prompts

## Project Status

Working prototype  
Under active development  
Research-oriented (not production-ready)

## Pipeline Overview

The pipeline executed by `main.py` follows this order:

1. **Video → Images**
   - Extract frames from videos
   - Frame skipping & resizing

2. **Pre-SAM Processing**
   - Filter and prepare images for SAM3

3. **SAM3 Segmentation**
   - Text-prompt-based segmentation
   - Convert boxes → YOLO format

4. **Post-SAM Cleaning**
   - Remove empty or invalid labels

5. **Data Augmentation**
   - Colour augmentation (hue, saturation, brightness)
   - Dust / noise simulation
   - Image distortions (labels preserved)

6. **Dataset Splitting**
   - Train / validation / test folders (YOLO format)

7. **YOLO Training**
   - Train YOLOv8 on generated dataset


## Requirements

- **Linux or WSL (strongly recommended)**
- NVIDIA GPU with CUDA support
- Python **3.10+**
- Conda / Miniconda
- Git

Windows-native Python is **not supported**.

### 1. Clone this repository

```bash
git clone https://github.com/A-makarim/Mars-Probe-Detection.git
cd Mars-Probe-Detection
```
From here, follow SAM3 README and clone it in the current directory

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e ".[train]"
cd ..
```
Follow the official SAM3 README to:
Request checkpoint access
Authenticate with HuggingFace
Download model weights

Install YOLO Ultralytics
Follow YOLO Ultralytics repository on GitHub for detailed setup
```bash
pip install ultralytics
```

## Run Full Pipeline
```bash
python main.py
```
