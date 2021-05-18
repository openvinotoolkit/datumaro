class PointCloudPath:
    DEFAULT_DIR = "ds0"
    ANNNOTATION_DIR = "ann"

    IMAGE_EXT = ".jpg"
    FILE_EXT = ".json"

    POINT_CLOUD_DIR = "pointcloud"
    RELATED_IMAGES = "related_images"

    WRITE_FILES = ["meta.json", "key_id_map.json"]

    BUILTIN_ATTRS = ["occluded", "source", "group"]
