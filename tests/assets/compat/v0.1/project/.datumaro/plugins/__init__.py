from datumaro.components.extractor import DatasetItem, SubsetBase


class MyExtractor(SubsetBase):
    def __iter__(self):
        yield from [
            DatasetItem("1"),
            DatasetItem("2"),
        ]
