Visualizer
##########

This API allows you to visualize a dataset with media ids.

Although Datumaro supports various kinds of vision tasks, e.g., classification, object detection,
semantic segmentation, key point estimation, visual captioning, etc., we provide a task-agnostic
visualization tool. That is, regardless of annotation types, `vis_gallery` describes the multiple
annotation-overlapped images from a list of multiple media ids. We can control the transparency of
annotations over images by adjusting `alpha`.

.. toctree::
   :maxdepth: 1
   :caption: Jupyter Notebook Example:

   notebooks/03_visualize
