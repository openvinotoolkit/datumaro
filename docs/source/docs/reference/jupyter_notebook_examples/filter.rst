filter
######

This API allows you to filter a dataset to satisfy some conditions.
Here, XML `XPath <https://devhints.io/xpath>`_ is used as a query format.

For instance, with a given XML file below, we can filter a dataset by the subset name through
`/item[subset="minival2014"]`, by the media id through `/item[id="290768"]`, by the image sizes
through `/item[image/width=image/height]`, and annotation information such as id (`id`), type
(`type`), label (`label_id`), bounding box (`x, y, w, h`), etc.

.. code-block::

    <item>
      <id>290768</id>
      <subset>minival2014</subset>
      <image>
        <width>612</width>
        <height>612</height>
        <depth>3</depth>
      </image>
      <annotation>
        <id>80154</id>
        <type>bbox</type>
        <label_id>39</label_id>
        <x>264.59</x>
        <y>150.25</y>
        <w>11.19</w>
        <h>42.31</h>
        <area>473.87</area>
      </annotation>
      <annotation>
        <id>669839</id>
        <type>bbox</type>
        <label_id>41</label_id>
        <x>163.58</x>
        <y>191.75</y>
        <w>76.98</w>
        <h>73.63</h>
        <area>5668.77</area>
      </annotation>
      ...
    </item>

For the annotation-based filtration, we need to set the argument `filter_annotations` to `True`.
We provide the argument `remove_empty` to remove all media with an empty annotation. We note that
datasets are updated in-place by default.

.. toctree::
   :maxdepth: 1
   :caption: Jupyter Notebook Example:

   notebooks/04_filter
