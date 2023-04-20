Datumaro module
===============
.. _datumaro:

The recommended way to use Datumaro API is:

.. code-block:: python

    import datumaro as dm

Once you can use the components:

* :mod:`dm.annotation <datumaro.components.annotation>`
    * :attr:`dm.NO_GROUP <datumaro.components.annotation.NO_GROUP>`
    * :class:`dm.Annotation <datumaro.components.annotation.Annotation>`
    * :class:`dm.AnnotationType <datumaro.components.annotation.AnnotationType>`
    * :class:`dm.Bbox <datumaro.components.annotation.Bbox>`
    * :attr:`dm.BinaryMaskImage <datumaro.components.annotation.BinaryMaskImage>`
    * :class:`dm.Caption <datumaro.components.annotation.Caption>`
    * :class:`dm.Categories <datumaro.components.annotation.Categories>`
    * :class:`dm.Colormap <datumaro.components.annotation.Colormap>`
    * :class:`dm.CompiledMask <datumaro.components.annotation.CompiledMask>`
    * :class:`dm.CompiledMaskImage <datumaro.components.annotation.CompiledMaskImage>`
    * :class:`dm.Cuboid3d <datumaro.components.annotation.Cuboid3d>`
    * :class:`dm.IndexMaskImage <datumaro.components.annotation.IndexMaskImage>`
    * :class:`dm.Label <datumaro.components.annotation.Label>`
    * :class:`dm.LabelCategories <datumaro.components.annotation.LabelCategories>`
    * :class:`dm.Mask <datumaro.components.annotation.Mask>`
    * :class:`dm.MaskCategories <datumaro.components.annotation.MaskCategories>`
    * :class:`dm.Points <datumaro.components.annotation.Points>`
    * :class:`dm.PointsCategories <datumaro.components.annotation.PointsCategories>`
    * :class:`dm.Polygon <datumaro.components.annotation.Polygon>`
    * :class:`dm.PolyLine <datumaro.components.annotation.PolyLine>`
    * :class:`dm.RgbColor <datumaro.components.annotation.RgbColor>`
    * :class:`dm.RleMask <datumaro.components.annotation.RleMask>`

* :mod:`dm.cli_plugin <datumaro.components.cli_plugin>`
    * :class:`dm.CliPlugin <datumaro.components.cli_plugin.CliPlugin>`

.. config_model
.. config_model
.. crypter
.. dataset_base
.. dataset
.. environment
.. errors
.. exporter
.. extractor_tfds
.. filter
.. format_detection
.. generator
.. hl_ops
.. Importer
.. launcher
.. media_manager
.. media
.. merger
.. operations
.. progress_reporting
.. project
.. explorer
.. shift_analyzer
.. transformer
.. validator
.. visualizer

* :mod:`dm.errors <datumaro.components.errors>`

* :mod:`dm.ops <datumaro.components.operations>`

* :mod:`dm.project <datumaro.components.project>`


* :mod:`dm.dataset <datumaro.components.dataset>`
    * :class:`dm.Dataset <datumaro.components.dataset.Dataset>`
    * :class:`dm.DatasetPatch <datumaro.components.dataset.DatasetPatch>`
    * :class:`dm.DatasetSubset <datumaro.components.dataset.DatasetSubset>`
    * :class:`dm.IDataset <datumaro.components.dataset.IDataset>`
    * :class:`dm.ItemStatus <datumaro.components.dataset.ItemStatus>`
    * :func:`dm.eager_mode <datumaro.components.dataset.eager_mode>`

* :mod:`dm.environment <datumaro.components.environment>`
    * :class:`dm.Environment <datumaro.components.environment.Environment>`
    * :class:`dm.PluginRegistry <datumaro.components.environment.PluginRegistry>`

* :mod:`dm.extractor <datumaro.components.extractor>`
    * :class:`dm.DEFAULT_SUBSET_NAME <datumaro.components.extractor.DEFAULT_SUBSET_NAME>`
    * :class:`dm.CategoriesInfo <datumaro.components.extractor.CategoriesInfo>`
    * :class:`dm.DatasetItem <datumaro.components.extractor.DatasetItem>`
    * :class:`dm.Extractor <datumaro.components.extractor.Extractor>`
    * :class:`dm.IExtractor <datumaro.components.extractor.IExtractor>`
    * :class:`dm.Importer <datumaro.components.extractor.Importer>`
    * :class:`dm.ItemTransform <datumaro.components.extractor.ItemTransform>`
    * :class:`dm.SourceExtractor <datumaro.components.extractor.SourceExtractor>`
    * :class:`dm.Transform <datumaro.components.extractor.Transform>`

* :class:`dm.hl_ops <datumaro.components.hl_ops>`
    * :func:`dm.export <datumaro.components.hl_ops.export>`
    * :func:`dm.filter <datumaro.components.hl_ops.filter>`
    * :func:`dm.merge <datumaro.components.hl_ops.merge>`
    * :func:`dm.run_model <datumaro.components.hl_ops.run_model>`
    * :func:`dm.transform <datumaro.components.hl_ops.transform>`
    * :func:`dm.validate <datumaro.components.hl_ops.validate>`

* :mod:`dm.launcher <datumaro.components.launcher>`
    * :class:`dm.Launcher <datumaro.components.launcher.Launcher>`
    * :class:`dm.ModelTransform <datumaro.components.launcher.ModelTransform>`

* :mod:`dm.media <datumaro.components.media>`
    * :class:`dm.ByteImage <datumaro.components.media.ByteImage>`
    * :class:`dm.Image <datumaro.components.media.Image>`
    * :class:`dm.MediaElement <datumaro.components.media.MediaElement>`
    * :class:`dm.Video <datumaro.components.media.Video>`
    * :class:`dm.VideoFrame <datumaro.components.media.VideoFrame>`

* :mod:`dm.media_manager <datumaro.components.media_manager>`
    * :class:`dm.MediaManager <datumaro.components.media_manager.MediaManager>`

* :mod:`dm.validator <datumaro.components.validator>`
    * :class:`dm.Validator <datumaro.components.validator.Validator>`
