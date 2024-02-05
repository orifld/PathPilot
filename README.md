## Path Pilot
Path Pilot project is an application that allows real-time analysis of the surroundings of a blind person.
Using voice commands, it tells the person which obstacles are on his way and how can he bypass them.

Path Pilot was built in Python, combining [ultralytics YOLOv8] (https://github.com/ultralytics/ultralytics) for objects segmentation estimation
and [MiDaS] (https://github.com/isl-org/MiDaS) for image depth prediction.

## Screenshots
Screenshot #1

<p align="center">
<a href="https://ibb.co/k9txGnS"><img style="max-width:200px; width:90%"  src="https://i.ibb.co/2sTyF2k/Figure-2024-02-04-200710.png" alt="Path-Pilot-Screenshot-1" ></a>
</p>

Screenshot #2

<p align="center">
<a href="https://ibb.co/s5DcbHy"><img style="max-width:200px; width:90%"  src="https://i.ibb.co/hXnrLfD/Figure-2024-02-04-200800.png" alt="Path-Pilot-Screenshot-2" ></a>
</p>

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
python Main.py --video_source_path './Videos/GH012163_640_2FPS.mp4'
```
