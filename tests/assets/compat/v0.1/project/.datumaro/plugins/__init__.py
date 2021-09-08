from datumaro.components.extractor import DatasetItem, SourceExtractor


class MyExtractor(SourceExtractor):
    def __iter__(self):
        yield from [
            DatasetItem('1'),
            DatasetItem('2'),
        ]
