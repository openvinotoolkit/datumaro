---
title: 'MARS'
linkTitle: 'MARS'
description: ''
weight: 14
---

## Format specification

MARS is a dataset for and motion analysis and person identification task,
and this dataset it's extension of Market-1501 dataset format.
MARS dataset is available for downloading
[here](http://zheng-lab.cecs.anu.edu.au/Project/project_mars.html)

Supported types of annotations:
- `Label`

Required attributes:
- `is_distractors` (bool): True when image with distractors,
    which negatively affect retrieval accuracy
- `is_junk`: True for junk image which do not affect retrieval accuracy;
- `pedestrian_id`: four-digit number in format `%04d`;
- `camera_id`: one-digit number;
- `track_id`: four-digit number in format `%04d`;
- `frame_id`: three-digit number in format `%03d`, that mean number of
  frame within this track. For the tracks, their names are accumulated
  for each ID, but for frames, they start from "0001" in each track.


## Import MARS dataset

Use these instructions to import MARS dataset into Datumaro project:

```bash
datum create
datum add -f mars ./dataset
```

> Note: the directory with dataset should be subdirectory of the
> project directory.

```
mars_dataset
├── <bbox_subset_name1>
│   ├── 0001 # directory with images of pedestrian with id 0001
│   │   ├── 0001C1T0001F001.jpg
│   │   ├── 0001C1T0001F002.jpg
│   │   ├── ...
│   ├── 0002 # directory with images of pedestrian with id 0002
│   │   ├── 0002C1T0001F001.jpg
│   │   ├── 0002C1T0001F001.jpg
│   │   ├── ...
│   ├── 0000 # distractors images, which negatively affect retrieval accuracy.
│   │   ├── 0000C1T0001F001.jpg
│   │   ├── 0000C1T0001F001.jpg
│   │   ├── ...
│   ├── 00-1 # junk images which do not affect retrieval accuracy
│   │   ├── 00-1C1T0001F001.jpg
│   │   ├── 00-1C1T0001F001.jpg
│   │   ├── ...
├── <bbox_subset_name2>
│   ├── ...
├── ...
```

All images in MARS dataset has strict convention of naming:
```
xxxxCxTxxxxFxxx.jpg
```
- the first four digits indicate the pedestrian's number;
- digits after `C` indicate the camera id;
- four digits after `T` indicate the track id for this pedestrian;
- three digits after `F` indicate the frame id with this track

> Note: there are two specific pedestrian IDs 0000 and 00-1
> which indicate distracting images and unwanted images respectively
