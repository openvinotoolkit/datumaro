from unittest import TestCase
import os.path as osp

from datumaro.components.config import Config, DictConfig, SchemaBuilder
from datumaro.util.test_utils import TestDir

from .requirements import Requirements, mark_requirement


class ConfigTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_produce_multilayer_config_from_dict(self):
        schema_low = SchemaBuilder() \
            .add('options', dict) \
            .build()
        schema_mid = SchemaBuilder() \
            .add('desc', lambda: Config(schema=schema_low)) \
            .build()
        schema_top = SchemaBuilder() \
            .add('container', lambda: DictConfig(
                lambda v: Config(v, schema=schema_mid))) \
            .build()

        value = 1
        source = Config({
            'container': {
                'elem': {
                    'desc': {
                        'options': {
                            'k': value
                        }
                    }
                }
            }
        }, schema=schema_top)

        self.assertEqual(value, source.container['elem'].desc.options['k'])

    def test_can_save_and_load(self):
        with TestDir() as test_dir:
            schema_low = SchemaBuilder() \
                .add('options', dict) \
                .build()
            schema_mid = SchemaBuilder() \
                .add('desc', lambda: Config(schema=schema_low)) \
                .build()
            schema_top = SchemaBuilder() \
                .add('container', lambda: DictConfig(
                    lambda v: Config(v, schema=schema_mid))) \
                .build()

            source = Config({
                'container': {
                    'elem': {
                        'desc': {
                            'options': {
                                'k': (1, 2, 3),
                                'd': 'asfd',
                            }
                        }
                    }
                }
            }, schema=schema_top)
            p = osp.join(test_dir, 'f.yaml')

            source.dump(p)

            loaded = Config.parse(p, schema=schema_top)

            self.assertTrue(isinstance(
                loaded.container['elem'].desc.options['k'], list))
            loaded.container['elem'].desc.options['k'] = \
                tuple(loaded.container['elem'].desc.options['k'])
            self.assertEqual(source, loaded)
