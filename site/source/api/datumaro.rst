Datumaro
########

For external use, the recommended way to call Datumaro API is:

.. code-block:: python

    import datumaro as dm

Once you can use the components:

* **dm.annotation** :mod:`datumaro.components.annotation`
    * **dm.NO_GROUP**
    * **dm.Annotation** :class:`datumaro.components.annotation.Annotation`
    * **dm.AnnotationType** :class:`datumaro.components.annotation.AnnotationType`
    * **dm.Bbox** :class:`datumaro.components.annotation.Bbox`
    * **dm.BinaryMaskImage**
    * **dm.Caption** :class:`datumaro.components.annotation.Caption`
    * **dm.Categories** :class:`datumaro.components.annotation.Categories`
    * **dm.Colormap** :class:`datumaro.components.annotation.Colormap`
    * **dm.CompiledMask** :class:`datumaro.components.annotation.CompiledMask`
    * **dm.CompiledMaskImage**
    * **dm.Cuboid3d** :class:`datumaro.components.annotation.Cuboid3d`
    * **dm.IndexMaskImage**
    * **dm.Label** :class:`datumaro.components.annotation.Label`
    * **dm.LabelCategories** :class:`datumaro.components.annotation.LabelCategories`
    * **dm.Mask** :class:`datumaro.components.annotation.Mask`
    * **dm.MaskCategories** :class:`datumaro.components.annotation.MaskCategories`
    * **dm.Points** :class:`datumaro.components.annotation.Points`
    * **dm.PointsCategories** :class:`datumaro.components.annotation.PointsCategories`
    * **dm.Polygon** :class:`datumaro.components.annotation.Polygon`
    * **dm.PolyLine** :class:`datumaro.components.annotation.PolyLine`
    * **dm.RgbColor**
    * **dm.RleMask** :class:`datumaro.components.annotation.RleMask`

* **dm.errors** :mod:`datumaro.components.errors`

* **dm.operations** :mod:`datumaro.components.operations`

* **dm.project** :mod:`datumaro.components.project`

* **dm.cli_plugin** :mod:`datumaro.components.cli_plugin`
    * **dm.CliPlugin** :class:`datumaro.components.cli_plugin.CliPlugin`

* **dm.converter** :mod:`datumaro.components.converter`
    * **dm.Converter** :class:`datumaro.components.converter.Converter`

* **dm.dataset** :mod:`datumaro.components.dataset`
    * **dm.Dataset** :class:`datumaro.components.dataset.Dataset`
    * **dm.DatasetPatch** :class:`datumaro.components.dataset.DatasetPatch`
    * **dm.DatasetSubset** :class:`datumaro.components.dataset.DatasetSubset`
    * **dm.IDataset**
    * **dm.ItemStatus** :class:`datumaro.components.dataset.ItemStatus`
    * **dm.eager_mode** :func:`datumaro.components.dataset.eager_mode`

* **dm.environment** :mod:`datumaro.components.environment`
    * **dm.Environment** :class:`datumaro.components.environment.Environment`
    * **dm.PluginRegistry** :class:`datumaro.components.environment.PluginRegistry`

* **dm.extractor** :mod:`datumaro.components.extractor`
    * **dm.DEFAULT_SUBSET_NAME**
    * **dm.CategoriesInfo**
    * **dm.DatasetItem** :class:`datumaro.components.extractor.DatasetItem`
    * **dm.Extractor** :class:`datumaro.components.extractor.Extractor`
    * **dm.IExtractor** :class:`datumaro.components.extractor.IExtractor`
    * **dm.Importer** :class:`datumaro.components.extractor.Importer`
    * **dm.ItemTransform** :class:`datumaro.components.extractor.ItemTransform`
    * **dm.SourceExtractor** :class:`datumaro.components.extractor.SourceExtractor`
    * **dm.Transform** :class:`datumaro.components.extractor.Transform`

* **dm.hl_ops** :class:`datumaro.components.hl_ops`
    * **dm.export** :func:`datumaro.components.hl_ops.export`
    * **dm.filter** :func:`datumaro.components.hl_ops.filter`
    * **dm.merge** :func:`datumaro.components.hl_ops.merge`
    * **dm.run_model** :func:`datumaro.components.hl_ops.run_model`
    * **dm.transform** :func:`datumaro.components.hl_ops.transform`
    * **dm.validate** :func:`datumaro.components.hl_ops.validate`

* **dm.launcher** :mod:`datumaro.components.launcher`
    * **dm.Launcher** :class:`datumaro.components.launcher.Launcher`
    * **dm.ModelTransform** :class:`datumaro.components.launcher.ModelTransform`

* **dm.media** :mod:`datumaro.components.media`
    * **dm.ByteImage** :class:`datumaro.components.media.ByteImage`
    * **dm.Image** :class:`datumaro.components.media.Image`
    * **dm.MediaElement** :class:`datumaro.components.media.MediaElement`
    * **dm.Video** :class:`datumaro.components.media.Video`
    * **dm.VideoFrame** :class:`datumaro.components.media.VideoFrame`

* **dm.media_manager** :mod:`datumaro.components.media_manager`
    * **dm.MediaManager** :class:`datumaro.components.media_manager.MediaManager`

* **dm.validator** :mod:`datumaro.components.validator`
    * **dm.Validator** :class:`datumaro.components.validator.Validator`
