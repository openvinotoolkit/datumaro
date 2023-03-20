---
title: 'Video'
linkTitle: 'Video'
description: ''
---

## Format specification
There are many kinds of video extensions as listed up
[here](https://github.com/openvinotoolkit/datumaro/blob/develop/datumaro/plugins/data_formats/video.py).

Datumaro can import a video into video frames by adjusting the start frame, end frame,
and step size. Furthermore, with a `video_keyframes` format, Datumaro can extract
keyframes by comparing zero-mean normalized cross correlation (ZNCC) metric between
successive frames as following [here](https://www.sciencedirect.com/science/article/pii/S1047320312000223).
Plus, Datumaro provides the options for choosing image extension and name patterns
for efficient data management from multiple videos.

## Import video

A Datumaro project with a video frames can be created
in the following way:

```
datum create
datum import -f video_frames <path_to_video>
```

Load video through the Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path_to_video>', format='video_frames')
```

Datumaro has few import options for `video_frames` format, to apply them
use the `--` after the main command argument.

`video_frames` import options:
- `-i, --input-path` (string) - Path to the video file
- `-o, --output-dir` (string) - Output directory. By default, a subdirectory
  in the current directory is used
- `--overwrite` - Allows overwriting existing files in the output directory,
  when it is not empty
- `-p, --name-pattern` (string) - Name pattern for the produced
  images (default: `%06d`)
- `-s, --step` (integer) - Frame step (default: 1)
- `-b, --start-frame` (integer) - Starting frame (default: 0)
- `-e, --end-frame` (integer) - Finishing frame (default: none)
- `-x, --image-ext` (string) Output image extension (default: `.jpg`)
- `-h, --help` - Print the help message and exit

Usage:

``` bash
datum import -f video_frames [-h] SRC_PATH [-o DST_DIR] [--overwrite]
  [-n NAME_PATTERN] [-s STEP] [-b START_FRAME] [-e END_FRAME] [-x IMAGE_EXT]
```

Example: import a video into frames, use each 30-rd frame:
```bash
datum import -f video_frames video.mp4 -o video-frames --step 30
```

Example: import a video into frames, save as 'frame_xxxxxx.png' files:
```bash
datum import -f video_frames video.mp4 -o video-frames --image-ext=.png --name-pattern='frame_%%06d'
```

Example: import a video into keyframes:
```bash
datum import -f video_keyframes video.mp4 -o video-frames
```
