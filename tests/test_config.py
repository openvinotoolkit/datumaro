from unittest import TestCase

from datumaro.components.config import Config, DictConfig, SchemaBuilder

import pytest
from tests.requirements import Requirements
from tests.requirements import DatumaroComponent


@pytest.mark.components(DatumaroComponent.Datumaro)
class ConfigTest(TestCase):
    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.component
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
