# napari-cellular-automata

[![License GNU GPL v3.0](https://img.shields.io/pypi/l/napari-cellular-automata.svg?color=green)](https://github.com/brisvag/napari-cellular-automata/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-cellular-automata.svg?color=green)](https://pypi.org/project/napari-cellular-automata)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-cellular-automata.svg?color=green)](https://python.org)
[![tests](https://github.com/brisvag/napari-cellular-automata/workflows/tests/badge.svg)](https://github.com/brisvag/napari-cellular-automata/actions)
[![codecov](https://codecov.io/gh/brisvag/napari-cellular-automata/branch/main/graph/badge.svg)](https://codecov.io/gh/brisvag/napari-cellular-automata)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-cellular-automata)](https://napari-hub.org/plugins/napari-cellular-automata)

A generalized n-dimensional cellular automata engine, using [napari](https://github.com/napari/napari) for visualisation.

https://user-images.githubusercontent.com/23482191/235496634-e4d56432-77dc-4acf-838a-61c98d32c1c0.mp4

## Installation and Usage

```
pip install napari-cellular-automata
```

Open `napari`, then use the `Initialize World` widget to create a new layer with the automaton, of your choice. Use `Populate World` to randomize the initial state as you prefer. Then use the `Run Automaton` widget to run the simulation. You can select multiple states to plot some stats as well.

---

You can also use this as a library:

```python
from napari-cellular-automata.engine import populate_world_with_state, step_world, get_state_descriptions
from napari-cellular-automata.automata import AUTOMATA
import numpy as np

pyro = AUTOMATA['Pyroclastic']
world = np.zeros((100, 100, 100), dtype=np.uint8)
populate_world_with_state(world, 9, blob_number=1)
states = get_state_descriptions(pyro)

while True:
    uniq, counts = np.unique(world, return_counts=True)
    for state, count in zip(uniq, counts):
        print(f'{states[state]}: {count}')
    world = step_world(world, pyro, wrap=False, edge_value=10)
```

## Rules and engines

In the generalized approach, rules are defined by transition dictionaries that encode how each state changes into a different state, plus some extra info such as color and state name for visualisation. This implementation is based on a [series of videos](https://www.youtube.com/watch?v=ygdPRlSo3Qg) by [tsoding](https://github.com/tsoding/). Original typescript source code: https://github.com/tsoding/autocell.

For example, [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) can be defined as follows:

```python
[
    { # state 0 (dead)
        'transitions': {
            # if this cell has 3 neighbours of state 1 (alive), become state 1
            # -1 means "any number" (only relevant with more than 2 states)
            (-1, 3): 1,
        },
        # if none of the rules above apply, become/remain the "default" state 0 (dead)
        'default': 0,
        # visualisation parameters
        'color': 'transparent',
        'name': 'dead',
    },
    { # state 1 (alive)
        'transitions': {
            # if this cell has 2 or 3 neighbours of state 1 (alive), remain state 1
            (-1, 2): 1,
            (-1, 3): 1,
        },
        # if none of the rules above apply, become the "default" state 0 (dead)
        'default': 0,
        'color': 'green',
        'name': 'alive',
    },
]
```

---

A second, more limited (but smaller and faster) engine is based on the [Survival-Birth](https://softologyblog.wordpress.com/2019/12/28/3d-cellular-automata-3/) rules; neighbour types do not matter, just their number. Other than dead and alive, any number of decay states can be added, which simply "delay" the time of death of a cell. The `Game of Life` is defined as:

```python
{
    # number of non-dead neighbours necessary for survival
    'survival': [2, 3],
    # number of non-dead neighbours necessary for birth
    'birth': [3],
    # number of total states (including alive, dead, and decay)
    'states': 2,
}
```

or, in short form: `'2-3/3/2'`.


You can add a custom automaton by simply adding to the `AUTOMATA` dictionary:

```python
from napari-cellular-automata.engine import rules_from_survival_birth
from napari-cellular-automata.automata import AUTOMATA

AUTOMATA['My rules'] = {
    'ndim': 2,
    'mode': 'survival-birth',
    'rules': rules_from_survival_birth('2-3,5/3/4'),
}
```


----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-cellular-automata` via [pip]:

    pip install napari-cellular-automata



To install latest development version :

    pip install git+https://github.com/brisvag/napari-cellular-automata.git


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [GNU GPL v3.0] license,
"napari-cellular-automata" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/brisvag/napari-cellular-automata/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
