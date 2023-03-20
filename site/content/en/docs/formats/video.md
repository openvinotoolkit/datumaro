---
title: 'Video'
linkTitle: 'Video'
description: ''
---

## Format specification
There are many kinds of video extensions as listed up [here]
(https://github.com/openvinotoolkit/datumaro/blob/develop/datumaro/plugins/data_formats/video.py).

Datumaro can import the video data and The original ImageNet dataset contains about 1.2M images and information
about class name for each image. Datumaro supports two versions of ImageNet
format: `imagenet` and `imagenet_txt`. The `imagenet_txt` format assumes storing
information about the class of the image in `*.txt` files. And `imagenet` format
assumes storing information about the class of the image in the name of
directory where is this image stored.

## Import Video

A Datumaro project with a video frames can be created
in the following way:

```
datum create
datum import -f imagenet <path_to_dataset>
```

> Note: if you use `datum import` then <path_to_dataset> should not be a
> subdirectory of directory with Datumaro project, see more information about
> it in the [docs](/docs/user-manual/command-reference/sources/#source-add).

Load ImageNet dataset through the Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path_to_dataset>', format='imagenet_txt')
```

For successful importing of ImageNet dataset the input directory with dataset
should has the following structure:

<!--lint disable fenced-code-flag-->
{{< tabpane >}}
  {{< tab header="imagenet">}}
imagenet_dataset/
├── label_0
│   ├── <image_name_0>.jpg
│   ├── <image_name_1>.jpg
│   ├── <image_name_2>.jpg
│   ├── ...
├── label_1
│    ├── <image_name_0>.jpg
│    ├── <image_name_1>.jpg
│    ├── <image_name_2>.jpg
│    ├── ...
├── ...
  {{< /tab >}}
  {{< tab header="imagenet_txt">}}
imagenet_txt_dataset/
├── images # directory with images
│   ├── <image_name_0>.jpg
│   ├── <image_name_1>.jpg
│   ├── <image_name_2>.jpg
│   ├── ...
├── synsets.txt # optional, list of labels
└── train.txt   # list of pairs (image_name, label)
  {{< /tab >}}
{{< /tabpane >}}

> Note: if you don't have synsets file then Datumaro will automatically generate
> classes with a name pattern `class-<i>`.

Datumaro has few import options for `imagenet_txt` format, to apply them
use the `--` after the main command argument.

`imagenet_txt` import options:
- `--labels` {`file`, `generate`}: allow to specify where to get label
  descriptions from (use `file` to load from the file specified
  by `--labels-file`; `generate` to create generic ones)
- `--labels-file` allow to specify path to the file with label descriptions
  ("synsets.txt")

Usage:

``` bash
datum import -f video_frames [-h] SRC_PATH [-o DST_DIR] [--overwrite]
  [-n NAME_PATTERN] [-s STEP] [-b START_FRAME] [-e END_FRAME] [-x IMAGE_EXT]
```

Parameters:
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

Example: import a video into frames, use each 30-rd frame:
```bash
datum import -f video_frames video.mp4 -o video-frames --step 30
```

Example: import a video into frames, save as 'frame_xxxxxx.png' files:
```bash
datum import -f video_frames video.mp4 -o video-frames --image-ext=.png --name-pattern='frame_%%06d'
```

Example: import a video into keyfraems:
```bash
datum import -f video_keyframes video.mp4 -o video-frames --step 30
```
