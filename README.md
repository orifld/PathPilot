## Path Pilot

Path Pilot project is an application designed for real-time analysis of the surroundings for individuals with visual impairments. 
Utilizing voice commands, it provides information about obstacles in the user's path and suggests alternative routes to navigate them.

Developed in Python, Path Pilot leverages [ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object segmentation estimation 
and [MiDaS](https://github.com/isl-org/MiDaS) for image depth prediction.

## Setup 
Set up dependencies:

```shell
conda create -n PathPilot python=3.8
conda activate PathPilot
pip install ultralytics
pip install pygame
pip install gtts
pip install timm
```

## Usage

```shell
python Main.py --video_source_path [input_video_path]
```

## Screenshots
Screenshot #1

<p align="center">
<a href="https://ibb.co/k9txGnS"><img style="max-width:200px; width:90%"  src="https://i.ibb.co/2sTyF2k/Figure-2024-02-04-200710.png" alt="Path-Pilot-Screenshot-1" ></a>
</p>

Screenshot #2

<p align="center">
<a href="https://ibb.co/s5DcbHy"><img style="max-width:200px; width:90%"  src="https://i.ibb.co/hXnrLfD/Figure-2024-02-04-200800.png" alt="Path-Pilot-Screenshot-2" ></a>
</p>

