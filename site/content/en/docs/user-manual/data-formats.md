---
title: 'Supported data formats'
linkTitle: 'Data formats'
description: ''
weight: 4
tags: ["Formats"]
---

Datumaro only works with 2d RGB(A) images.

To create an unlabelled dataset from an arbitrary directory with images use
`ImageDir` format:

```bash
datum create -o <project/dir>
datum add path -p <project/dir> -f image_dir <directory/path/>
```

or if you work with Datumaro API:

For using with a project:

```python
from datumaro.components.project import Project

project = Project()
project.add_source('source1', {
  'format': 'image_dir',
  'url': 'directory/path/'
})
dataset = project.make_dataset()
```

And for using as a dataset:

```python
from datumaro.components.dataset import Dataset

dataset = Dataset.import_from('directory/path/', 'image_dir')
```

This will search for images in the directory recursively and add
them as dataset entries with names like `<subdir1>/<subsubdir1>/<image_name1>`.
The list of formats matches the list of supported image formats in OpenCV.
```
.jpg, .jpeg, .jpe, .jp2, .png, .bmp, .dib, .tif, .tiff, .tga, .webp, .pfm,
.sr, .ras, .exr, .hdr, .pic, .pbm, .pgm, .ppm, .pxm, .pnm
```

After addition into a project, images can be split into subsets and renamed
with transformations, filtered, joined with existing annotations etc.

To use a video as an input, one should either [create an Extractor plugin](/docs/developer-guide/plugins/),
which splits a video into frames, or split the video manually and import images.
