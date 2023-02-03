import os
import os.path as osp
from unittest import TestCase
from zipfile import ZipFile

import numpy as np

from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.media import Image
from datumaro.util.test_utils import TestDir, compare_datasets
from datumaro.util.test_utils import run_datum as run

from ...requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path


def make_zip_archive(src_path, dst_path):
    with ZipFile(dst_path, "w") as archive:
        for dirpath, _, filenames in os.walk(src_path):
            for name in filenames:
                path = osp.join(dirpath, name)
                archive.write(path, osp.relpath(path, src_path))


class ImageZipIntegrationScenarios(TestCase):
    @mark_requirement(Requirements.DATUM_267)
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", media=Image(data=np.ones((5, 5, 3)))),
                DatasetItem(id="2", media=Image(data=np.ones((2, 8, 3)))),
            ]
        )

        with TestDir() as test_dir:
            source_dataset.export(test_dir, format="image_dir")
            zip_path = osp.join(test_dir, "images.zip")
            make_zip_archive(test_dir, zip_path)

            proj_dir = osp.join(test_dir, "proj")
            run(self, "create", "-o", proj_dir)
            run(self, "import", "-p", proj_dir, "-f", "image_zip", zip_path)

            result_dir = osp.join(test_dir, "result")
            export_path = osp.join(result_dir, "export.zip")
            run(
                self,
                "export",
                "-p",
                proj_dir,
                "-f",
                "image_zip",
                "-o",
                result_dir,
                "--",
                "--name",
                osp.basename(export_path),
            )

            parsed_dataset = Dataset.import_from(export_path, format="image_zip")
            compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_267)
    def test_can_export_zip_images_from_coco_dataset(self):
        with TestDir() as test_dir:
            coco_dir = get_test_asset_path(
                "coco_dataset",
                "coco",
            )

            run(self, "create", "-o", test_dir)
            run(self, "import", "-p", test_dir, "-f", "coco", coco_dir)

            export_path = osp.join(test_dir, "export.zip")
            run(
                self,
                "export",
                "-p",
                test_dir,
                "-f",
                "image_zip",
                "-o",
                test_dir,
                "--overwrite",
                "--",
                "--name",
                osp.basename(export_path),
            )

            self.assertTrue(osp.isfile(export_path))
            with ZipFile(export_path, "r") as zf:
                images = {f.filename for f in zf.filelist}
                self.assertTrue(images == {"a.jpg", "b.jpg"})

    @mark_requirement(Requirements.DATUM_267)
    def test_can_change_extension_for_images_in_zip(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", media=Image(data=np.ones((5, 5, 3)))),
                DatasetItem(id="2", media=Image(data=np.ones((2, 8, 3)))),
            ]
        )

        with TestDir() as test_dir:
            source_dataset.export(test_dir, format="image_dir", image_ext=".jpg")
            zip_path = osp.join(test_dir, "images.zip")
            make_zip_archive(test_dir, zip_path)

            proj_dir = osp.join(test_dir, "proj")
            run(self, "create", "-o", proj_dir)
            run(self, "import", "-p", proj_dir, "-f", "image_zip", zip_path)

            export_path = osp.join(test_dir, "export.zip")
            run(
                self,
                "export",
                "-p",
                proj_dir,
                "-f",
                "image_zip",
                "-o",
                test_dir,
                "--overwrite",
                "--",
                "--name",
                osp.basename(export_path),
                "--image-ext",
                ".png",
            )

            self.assertTrue(osp.isfile(export_path))
            with ZipFile(export_path, "r") as zf:
                images = {f.filename for f in zf.filelist}
                self.assertTrue(images == {"1.png", "2.png"})
