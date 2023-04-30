from .engine import rules_from_survival_birth


AUTOMATA = {
    "Game of Life - full": {
        'ndim': 2,
        'mode': 'full',
        'rules': [
            {
                'transitions': {
                    (-1, 3): 1,
                },
                'default': 0,
                'color': 'transparent',
                'name': 'dead',
            },
            {
                'transitions': {
                    (-1, 2): 1,
                    (-1, 3): 1,
                },
                'default': 0,
                'color': 'green',
                'name': 'alive',
            },
        ],
    },
    "Game of Life - survival-birth": {
        'ndim': 2,
        'mode': 'survival-birth',
        'rules': {
            'survival': [2, 3],
            'birth': [3],
            'states': 2,
        },
    },
    # "Seeds": [
    #     {
    #         'transitions': {
    #             (-1, 2): 1,
    #         },
    #         'default': 0,
    #         'color': 'transparent',
    #         'name': 'dead',
    #     },
    #     {
    #         'default': 0,
    #         'color': 'green',
    #         'name': 'alive',
    #     },
    # ],
    # "Brian's Brain": [
    #     {
    #         'transitions': {
    #             (-1, 2, -1): 1,
    #         },
    #         'default': 0,
    #         'color': 'transparent',
    #         'name': 'dead',
    #     },
    #     {
    #         'default': 2,
    #         'color': 'green',
    #         'name': 'alive',
    #     },
    #     {
    #         'default': 0,
    #         'color': 'red',
    #         'name': 'dying',
    #     },
    # ],
    # "Wireworld": [
    #     {
    #         'default': 0,
    #         'color': 'transparent',
    #         'name': 'empty',
    #     },
    #     {
    #         'transitions': {
    #             (-1, -1, 1, -1): 2,
    #             (-1, -1, 2, -1): 2,
    #         },
    #         'default': 1,
    #         'color': 'yellow',
    #         'name': 'conductor',
    #     },
    #     {
    #         'default': 3,
    #         'color': 'blue',
    #         'name': 'electron head',
    #     },
    #     {
    #         'default': 1,
    #         'color': 'red',
    #         'name': 'electron tail',
    #     },
    # ],
}

AUTOMATA.update(
    {
        '445': rules_from_survival_birth('4/4/5'),
        'Pyroclastic': rules_from_survival_birth('4-7/6-8/10'),
    }
)
