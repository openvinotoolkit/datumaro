image module
------------

.. automodule:: datumaro.util.image

   .. autoclass:: _IMAGE_BACKENDS
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autofunction:: load_image

      NOTE: Check destination path for existence
      OpenCV silently fails if target directory does not exist

   .. autofunction:: save_image

   .. autofunction:: encode_image

   .. autofunction:: decode_image

   .. autofunction:: find_images

   .. autofunction:: is_image

   .. autoclass:: lazy_image
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autodata:: ImageMeta

   .. autofunction:: load_image_meta_file

   .. autofunction:: save_image_meta_file
