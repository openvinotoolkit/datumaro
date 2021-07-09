from unittest import TestCase
import os.path as osp

import numpy as np

from datumaro.util.image import (
    ByteImage, Image, encode_image, lazy_image, load_image,
    load_image_meta_file, save_image,
)
from datumaro.util.image_cache import ImageCache
from datumaro.util.test_utils import TestDir

from .requirements import Requirements, mark_requirement


class LazyImageTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_cache_works(self):
        with TestDir() as test_dir:
            image = np.ones((100, 100, 3), dtype=np.uint8)
            image_path = osp.join(test_dir, 'image.jpg')
            save_image(image_path, image)

            caching_loader = lazy_image(image_path, cache=None)
            self.assertTrue(caching_loader() is caching_loader())

            non_caching_loader = lazy_image(image_path, cache=False)
            self.assertFalse(non_caching_loader() is non_caching_loader())

class ImageCacheTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_cache_fifo_displacement(self):
        capacity = 2
        cache = ImageCache(capacity)

        loaders = [lazy_image(None, loader=lambda p: object(), cache=cache)
            for _ in range(capacity + 1)]

        first_request = [loader() for loader in loaders[1 : ]]
        loaders[0]() # pop something from the cache

        second_request = [loader() for loader in loaders[2 : ]]
        second_request.insert(0, loaders[1]())

        matches = sum([a is b for a, b in zip(first_request, second_request)])
        self.assertEqual(matches, len(first_request) - 1)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_global_cache_is_accessible(self):
        loader = lazy_image(None, loader=lambda p: object())

        ImageCache.get_instance().clear()
        self.assertTrue(loader() is loader())
        self.assertEqual(ImageCache.get_instance().size(), 1)

class ImageTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_lazy_image_shape(self):
        data = np.ones((5, 6, 3))

        image_lazy = Image(data=data, size=(2, 4))
        image_eager = Image(data=data)

        self.assertEqual((2, 4), image_lazy.size)
        self.assertEqual((5, 6), image_eager.size)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_ctors(self):
        with TestDir() as test_dir:
            path = osp.join(test_dir, 'path.png')
            image = np.ones([2, 4, 3])
            save_image(path, image)

            for args in [
                { 'data': image },
                { 'data': image, 'path': path },
                { 'data': image, 'path': path, 'size': (2, 4) },
                { 'data': image, 'path': path, 'loader': load_image, 'size': (2, 4) },
                { 'path': path },
                { 'path': path, 'loader': load_image },
                { 'path': 'somepath', 'loader': lambda p: image },
                { 'loader': lambda p: image },
                { 'path': path, 'size': (2, 4) },
            ]:
                with self.subTest(**args):
                    img = Image(**args)
                    # pylint: disable=pointless-statement
                    self.assertTrue(img.has_data)
                    self.assertEqual(img, image)
                    self.assertEqual(img.size, tuple(image.shape[:2]))
                    # pylint: enable=pointless-statement

class BytesImageTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_lazy_image_shape(self):
        data = encode_image(np.ones((5, 6, 3)), 'png')

        image_lazy = ByteImage(data=data, size=(2, 4))
        image_eager = ByteImage(data=data)

        self.assertEqual((2, 4), image_lazy.size)
        self.assertEqual((5, 6), image_eager.size)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_ctors(self):
        with TestDir() as test_dir:
            path = osp.join(test_dir, 'path.png')
            image = np.ones([2, 4, 3])
            image_bytes = encode_image(image, 'png')

            for args in [
                { 'data': image_bytes },
                { 'data': lambda _: image_bytes },
                { 'data': lambda _: image_bytes, 'ext': '.jpg' },
                { 'data': image_bytes, 'path': path },
                { 'data': image_bytes, 'path': path, 'size': (2, 4) },
                { 'data': image_bytes, 'path': path, 'size': (2, 4) },
                { 'path': path },
                { 'path': path, 'size': (2, 4) },
            ]:
                with self.subTest(**args):
                    img = ByteImage(**args)
                    # pylint: disable=pointless-statement
                    self.assertEqual('data' in args, img.has_data)
                    if img.has_data:
                        self.assertEqual(img, image)
                        self.assertEqual(img.get_bytes(), image_bytes)
                    img.size
                    if 'size' in args:
                        self.assertEqual(img.size, args['size'])
                    if 'ext' in args or 'path' in args:
                        self.assertEqual(img.ext, args.get('ext', '.png'))
                    # pylint: enable=pointless-statement

class ImageMetaTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_loading(self):
        meta_original = {
            'a': (123, 456),
            'b c': (10, 20),
        }

        with TestDir() as test_dir:
            meta_path = osp.join(test_dir, 'images.meta')

            with open(meta_path, 'w') as meta_file:
                for name, (h, w) in meta_original.items():
                    print(name, h, w, file=meta_file)

            meta_reloaded = load_image_meta_file(meta_path)

        self.assertEqual(meta_reloaded, meta_original)
