Media formats
=============

Datumaro supports the following media types:
- 2D RGB(A) images
- KITTI Point Clouds

To create an unlabelled dataset from an arbitrary directory with images use
``image_dir`` and ``image_zip`` formats:

.. code-block::

    datum create -o <project/dir>
    datum import -p <project/dir> -f image_dir <directory/path/>

or, if you work with Datumaro API:

- for using with a project:

.. code-block::

    from datumaro.project import Project

    project = Project.init()
    project.import_source('source1', format='image_dir', url='directory/path/')
    dataset = project.working_tree.make_dataset()

- for using as a dataset:

.. code-block::

    from datumaro import Dataset

    dataset = Dataset.import_from('directory/path/', 'image_dir')

This will search for images in the directory recursively and add
them as dataset entries with names like ``<subdir1>/<subsubdir1>/<image_name1>``.
The list of formats matches the list of supported image formats in OpenCV:
```
.jpg, .jpeg, .jpe, .jp2, .png, .bmp, .dib, .tif, .tiff, .tga, .webp, .pfm,
.sr, .ras, .exr, .hdr, .pic, .pbm, .pgm, .ppm, .pxm, .pnm
```

Once there is a ``Dataset`` instance, its items can be split into subsets,
renamed, filtered, joined with annotations, exported in various formats etc.

To import frames from a video, you can split the video into frames with
the [`split_video` command](/docs/user-manual/command-reference/util#split-video)
and then use the ``image_dir`` format described above. In more complex cases,
consider using  `FFmpeg <https://ffmpeg.org/>`_ and other tools for
video processing.

Alternatively, you can use the ``video_frames`` format directly:

.. note::

    However, that it can produce different results if the system
    environment changes. If you want to obtain reproducible results, consider
    splitting the video into frames by any method.

.. code-block::

    datum create -o <project/dir>
    datum import -p <project/dir> -f video_frames <video/path.avi>

.. code-block::

    from datumaro import Dataset

    dataset = Dataset.import_from('video.mp4', 'video_frames')

Datumaro supports the following video formats:
```
.3gp, .3g2, .asf, .wmv, .avi, .divx, .evo, .f4v, .flv, .mkv, .mk3d,
.mp4, .mpg, .mpeg, .m2p, .ps, .ts, .m2ts, .mxf, .ogg, .ogv, .ogx,
.mov, .qt, .rmvb, .vob, .webm
```
