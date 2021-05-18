import json
import os

from collections import OrderedDict
import os.path as osp


from datumaro.components.extractor import (SourceExtractor, DatasetItem,
    AnnotationType,Cuboid,
    LabelCategories, Importer
)

from datumaro.util.image import Image

from .format import PointCloudPath


class PointCloudExtractor(SourceExtractor):
    _SUPPORTED_SHAPES = "cuboid"
    mapping = {}
    meta = {}

    def __init__(self, path, subset=None):
        assert osp.isfile(path), path
        rootpath = osp.dirname(path)
        images_dir = ''
        if osp.isdir(osp.join(rootpath, PointCloudPath.ANNNOTATION_DIR)):
            images_dir = osp.join(rootpath, PointCloudPath.ANNNOTATION_DIR)
        self._images_dir = images_dir
        self._path = path

        if not subset:
            subset = osp.splitext(osp.basename(path))[0]

        super().__init__(subset=subset)

        items, categories = self._parse(path)
        self._items = list(self._load_items(items).values())
        self._categories = categories

    @classmethod
    def _parse(cls, path):
        path = osp.abspath(path)
        items = OrderedDict()
        categories = {}

        if osp.split(path)[-1] == "key_id_map.json":
            with open(path, "r") as f:
                data = f.read()
            cls.mapping = json.loads(data)

        meta_path = osp.abspath(osp.join(osp.dirname(path), "meta.json"))

        if osp.isfile(meta_path):

            with open(meta_path, "r") as f:
                meta = f.read()

            cls.meta = json.loads(meta)

        label_cat = LabelCategories()
        common_attrs = ["occluded"]

        if cls.meta:
            for label in cls.meta["classes"]:
                label_cat.add(label['title'], attributes=common_attrs)

        categories[AnnotationType.label] = label_cat

        data_dir = osp.join(osp.dirname(path), PointCloudPath.DEFAULT_DIR, PointCloudPath.ANNNOTATION_DIR)

        labels = {}
        for root, _, files in os.walk(data_dir):
            for file in files:
                with open(osp.join(data_dir, file), "r") as f:
                    figure_data = f.read()
                figure_data = json.loads(figure_data)

                for label in figure_data["objects"]:
                    labels.update({label["key"]: label["classTitle"]})

                group = 0
                z_order = 0
                attributes ={}

                for figure in figure_data["figures"]:
                    anno_points = []
                    geometry_type = ["position", "rotation", "dimensions"]
                    for geo in geometry_type:
                        [anno_points.append(float(i)) for i in figure["geometry"][geo].values()]

                    for i in range(7):
                        anno_points.append(0.0)

                    id = cls.mapping["figures"][figure['key']]
                    label = labels[figure["objectKey"]]

                    label = categories[AnnotationType.label].find(label)[0]

                    shape = Cuboid(anno_points, label=label, z_order=z_order,
                              id=id, attributes=attributes, group=group)

                    frame = cls.mapping["videos"][figure_data['key']]
                    frame_desc = items.get(frame, {'annotations': []})

                    frame_desc['annotations'].append(shape)
                    items[frame] = frame_desc


        return items, categories

    def _load_items(self, parsed):
        for frame_id, item_desc in parsed.items():
            name = item_desc.get('name', 'frame_%06d.png' % int(frame_id))
            image = osp.join(self._images_dir, name)
            image_size = (item_desc.get('height'), item_desc.get('width'))
            if all(image_size):
                image = Image(path=image, size=tuple(map(int, image_size)))

            parsed[frame_id] = DatasetItem(id=osp.splitext(name)[0],
                subset=self._subset, image=image,
                annotations=item_desc.get('annotations'),
                attributes={'frame': int(frame_id)})

        return parsed


class PointCloudImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '.json', 'point_cloud')
