annotation module
-----------------

.. automodule:: datumaro.components.annotation

   .. autoclass:: AnnotationType
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: NO_GROUP

   .. autoclass:: Annotation
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: Categories
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: LabelCategories
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: Label
      :members: __eq__, __init__
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      .. py:data:: label: int

   .. py:data:: RgbColor
   .. py:data:: Colormap

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
