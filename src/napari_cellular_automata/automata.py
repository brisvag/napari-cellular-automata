from .engine import rules_from_survival_birth

AUTOMATA = {
    "Game of Life - full": {
        "ndim": 2,
        "mode": "full",
        "rules": [
            {
                "transitions": {
                    (-1, 3): 1,
                },
                "default": 0,
                "color": "transparent",
                "name": "dead",
            },
            {
                "transitions": {
                    (-1, 2): 1,
                    (-1, 3): 1,
                },
                "default": 0,
                "color": "green",
                "name": "alive",
            },
        ],
    },
    "Game of Life - survival-birth": {
        "ndim": 2,
        "mode": "survival-birth",
        "rules": {
            "survival": [2, 3],
            "birth": [3],
            "states": 2,
        },
    },
    "Seeds": {
        "ndim": 2,
        "mode": "survival-birth",
        "rules": {
            "survival": [],
            "birth": [2],
            "states": 2,
        },
    },
    "Brian's Brain": {
        "ndim": 2,
        "mode": "full",
        "rules": [
            {
                "transitions": {
                    (-1, 2, -1): 1,
                },
                "default": 0,
                "color": "transparent",
                "name": "dead",
            },
            {
                "default": 2,
                "color": "green",
                "name": "alive",
            },
            {
                "default": 0,
                "color": "red",
                "name": "dying",
            },
        ],
    },
    "Wireworld": {
        "ndim": 2,
        "mode": "full",
        "rules": [
            {
                "default": 0,
                "color": "transparent",
                "name": "empty",
            },
            {
                "transitions": {
                    (-1, -1, 1, -1): 2,
                    (-1, -1, 2, -1): 2,
                },
                "default": 1,
                "color": "yellow",
                "name": "conductor",
            },
            {
                "default": 3,
                "color": "blue",
                "name": "electron head",
            },
            {
                "default": 1,
                "color": "red",
                "name": "electron tail",
            },
        ],
    },
    "445": {
        "ndim": 3,
        "mode": "survival-birth",
        "rules": rules_from_survival_birth("4/4/5"),
    },
    "Pyroclastic": {
        "ndim": 3,
        "mode": "survival-birth",
        "rules": rules_from_survival_birth("4-7/6-8/10"),
    },
    "Slow Decay": {
        "ndim": 3,
        "mode": "survival-birth",
        "rules": rules_from_survival_birth("1,4,8,11,13-26/13-26/5"),
    },
}
