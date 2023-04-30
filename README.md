# napari-cellular-automata

[![License GNU GPL v3.0](https://img.shields.io/pypi/l/napari-cellular-automata.svg?color=green)](https://github.com/brisvag/napari-cellular-automata/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-cellular-automata.svg?color=green)](https://pypi.org/project/napari-cellular-automata)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-cellular-automata.svg?color=green)](https://python.org)
[![tests](https://github.com/brisvag/napari-cellular-automata/workflows/tests/badge.svg)](https://github.com/brisvag/napari-cellular-automata/actions)
[![codecov](https://codecov.io/gh/brisvag/napari-cellular-automata/branch/main/graph/badge.svg)](https://codecov.io/gh/brisvag/napari-cellular-automata)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-cellular-automata)](https://napari-hub.org/plugins/napari-cellular-automata)

A generalized n-dimensional cellular automata engine, using @napari for visualisation.

Rules are defined as small "transition" dictionaries that define how each state changes into a different state, plus some extra info such as color and state name for visualisation.

For example, [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) is defined as follows:

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

Inspired by the series of videos by @tsoding starting with https://www.youtube.com/watch?v=ygdPRlSo3Qg. Original source code in https://github.com/tsoding/autocell.

Other references:
- https://softologyblog.wordpress.com/2019/12/28/3d-cellular-automata-3/

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
