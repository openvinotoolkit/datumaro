---
title: 'Media formats'
linkTitle: 'Media formats'
description: ''
weight: 5
---

Datumaro supports the following media types:
- 2D RGB(A) images
- KITTI Point Clouds

To create an unlabelled dataset from an arbitrary directory with images use
`image_dir` and `image_zip` formats:

``` bash
datum create -o <project/dir>
datum add -p <project/dir> -f image_dir <directory/path/>
```

or, if you work with Datumaro API:

- for using with a project:

  ```python
  from datumaro.components.project import Project

  project = Project.init()
  project.import_source('source1', format='image_dir', url='directory/path/')
  dataset = project.working_tree.make_dataset()
  ```

- for using as a dataset:

  ```python
  from datumaro.components.dataset import Dataset

  dataset = Dataset.import_from('directory/path/', 'image_dir')
  ```

This will search for images in the directory recursively and add
them as dataset entries with names like `<subdir1>/<subsubdir1>/<image_name1>`.
The list of formats matches the list of supported image formats in OpenCV:
```
.jpg, .jpeg, .jpe, .jp2, .png, .bmp, .dib, .tif, .tiff, .tga, .webp, .pfm,
.sr, .ras, .exr, .hdr, .pic, .pbm, .pgm, .ppm, .pxm, .pnm
```

Once there is a `Dataset` instance, its items can be split into subsets,
renamed, filtered, joined with annotations, exported in various formats etc.

To use a video as an input, one should either create a [plugin](/docs/user-manual/extending),
which splits a video into frames, or split the video manually and import images.
