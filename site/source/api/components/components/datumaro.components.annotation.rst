annotation module
-----------------

.. automodule:: datumaro.components.annotation

   .. autoclass:: AnnotationType
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autodata:: NO_GROUP

   .. autoclass:: Annotation
      :members: type, wrap, __eq__, __init__
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

         Describes an identifier of the annotation.
         Is not required to be unique within DatasetItem annotations or dataset.

            .. py:data:: id: int

         Arbitrary annotation-specific attributes. Typically, includes
         metainfo and properties that are not covered by other fields.
         If possible, try to limit value types of values by the simple
         builtin types (int, float, bool, str) to increase compatibility with
         different formats.
         There are some established names for common attributes like:
            - "occluded" (bool)
            - "visible" (bool)
         Possible dataset attributes can be described in Categories.attributes.

            .. py:data:: attributes: Dict[str, Any]

         Annotations can be grouped, which means they describe parts of a
         single object. The value of 0 means there is no group.

            .. py:data:: group: int

   .. autoclass:: Categories
      :members: __eq__, __init__
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

         Describes the list of possible annotation-type specific attributes in a dataset.

            .. py:data:: attributes: Set[str]

   .. autoclass:: LabelCategories
      :members: from_iterable, add, find, __getitem__, __contains__, __len__, __iter__, __eq__, __init__
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

         .. py:data:: items: List[str]

      .. py:class:: Category

         .. py:data:: name: str
                      parent: str
                      attributes: Set[str]

   .. autoclass:: Label
      :members: __eq__, __init__
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      .. py:data:: label: int

   .. autodata:: RgbColor
   .. autodata:: Colormap

   .. autoclass:: MaskCategories
      :members: inverse_colormap, __contains__, __getitem__, __len__, __eq__, __init__
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      .. automethod:: generate

         .. py:data:: colormap: Colormap


   .. autoclass:: BinaryMaskImage
   .. autoclass:: IndexMaskImage

   .. autoclass:: Mask
      :members: image, as_class_mask, as_instance_mask, get_area, get_bbox, paint, __eq__, __init__
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      .. py:data:: label: Optional[int]
                   z_order: int

   .. autoclass:: RleMask
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      .. py:data:: rle = attrib()

         uses pycocotools RLE representation

   .. autoclass:: CompiledMaskImage

   .. autoclass:: CompiledMask
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: _Shape
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: PolyLine
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      .. automethod:: _Shape.get_area

      .. automethod:: _Shape.get_bbox

   .. autoclass:: Cuboid3d
      :members: __init__, position, rotation, scale, __eq__
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      .. py:data:: label: Optional[int]

   .. autoclass:: Polygon
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      .. automethod:: _Shape.get_area(self)

      .. automethod:: _Shape.get_bbox(self)

   .. autoclass:: Bbox
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      .. automethod:: _Shape.get_area(self)

      .. automethod:: _Shape.get_bbox(self)

   .. autoclass:: PointsCategories
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      .. autoclass:: Category
         :members:
         :undoc-members:
         :private-members:
         :special-members:
         :show-inheritance:

               Names for specific points, e.g. eye, hose, mouth etc.
               These labels are not required to be in LabelCategories

            .. py:data:: labels: List[str]

               Pairs of connected point indices

            .. py:data:: joints: Set[Tuple[int, int]]

   .. autoclass:: Points
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      .. automethod:: _Shape.get_area(self)

      .. automethod:: _Shape.get_bbox(self)

      .. autoclass:: Visibility
         :members:
         :undoc-members:
         :show-inheritance:

   .. autoclass:: Caption
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:
