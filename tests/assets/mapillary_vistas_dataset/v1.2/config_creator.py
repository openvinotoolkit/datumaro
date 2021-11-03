import json

config = {
    'labels': [
        {
            'color': [
                10,
                50,
                90
            ],
            'instances': True,
            'readable': 'Bird',
            'name': 'animal--bird',
            'evaluate': True
        },
        {
            'color': [
                20,
                30,
                80
            ],
            'instances': True,
            'readable': 'Curb',
            'name': 'construction--barrier--curb',
            'evaluate': True
        },
        {
            'color': [
                30,
                70,
                40
            ],
            'instances': True,
            'readable': 'Person',
            'name': 'human--person',
            'evaluate': True
        },
    ],
    'version': '1.1',
    'mapping': 'public',
    'folder_structure': ''
}

with open('config_v1.2.json', 'w', encoding='utf-8') as f:
    json.dump(config, f)