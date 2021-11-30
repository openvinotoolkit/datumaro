converter module
----------------

.. automodule:: datumaro.components.converter

   .. autoclass:: Converter
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      .. automethod:: patch

         This solution is not any better in performance than just
         writing a dataset, but in case of patching (i.e. writing
         to the previous location), it allows to avoid many problems
         with removing and replacing existing files. Surely, this
         approach also has problems with removal of the given directory.
         Problems can occur if we can't remove the directory,
         or want to reuse the given directory. It can happen if it
         is mounted or (sym-)linked.
         Probably, a better solution could be to wipe directory
         contents and write new data there. Note that directly doing this
         also doesn't work, because images may be needed for writing.
