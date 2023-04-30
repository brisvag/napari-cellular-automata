#!/usr/bin/env python3

from time import sleep

from napari import Viewer
from napari.qt import thread_worker
from napari.layers import Labels
from magicgui import magic_factory


from .automata import AUTOMATA
from .engine import step_world, init_world, get_colors, get_state_descriptions


def _update_text_overlay(viewer):
    selected = viewer.layers.selection.active
    if selected is None:
        viewer.text_overlay.visible = False
        return

    auto = selected.metadata.get('automaton', None)
    if auto is None:
        viewer.text_overlay.visible = False
        return

    viewer.text_overlay.update(
        dict(
            visible=True,
            position='top_left',
            text=(', ').join(f"{name}: {desc}" for name, desc in get_state_descriptions(auto).items())
        )
    )


@magic_factory(
    call_button='Generate',
    automaton=dict(choices=AUTOMATA.keys()),
    x=dict(widget_type='Slider', min=10, max=1000),
    y=dict(widget_type='Slider', min=10, max=1000),
    z=dict(widget_type='Slider', min=0, max=1000),
)
def initialize_world(viewer: Viewer, automaton: str, random_ratios: str = '', x: int = 100, y: int = 100, z: int = 0) -> Labels:
    selection = viewer.layers.selection
    if _update_text_overlay not in selection.events.callbacks:
        selection.events.connect(lambda: _update_text_overlay(viewer))

    if not random_ratios:
        random_ratios = '1'
    ratios = [float(r) for r in random_ratios.split(',')]

    if z > 0:
        shape = (z, y, x)
    else:
        shape = (y, x)

    world = init_world(shape, ratios)

    auto = AUTOMATA[automaton]

    lb = Labels(
        world,
        name=automaton,
        color=get_colors(auto),
        opacity=1,
        metadata={'automaton': auto},
    )
    lb.brush_size = 1
    lb.bounding_box.visible = True
    lb.bounding_box.points = False
    lb.bounding_box.line_color = 'grey'

    return lb


@magic_factory(
    call_button='Run/Pause',
    fps=dict(widget_type='Slider', min=1, max=120),
)
def run_automaton(viewer: Viewer, layer: Labels, fps=60, wrap=True, edge_value=0):
    if getattr(run_automaton, '_worker', None) is not None:
        run_automaton._worker.quit()
        run_automaton._worker = None
        return

    selection = viewer.layers.selection
    if _update_text_overlay not in selection.events.callbacks:
        selection.events.connect(lambda: _update_text_overlay(viewer))

    def set_data(data):
        layer.data = data

    rules = layer.metadata['automaton']

    @thread_worker(connect={'yielded': set_data})
    def run_threaded():
        while True:
            yield step_world(layer.data, rules, wrap=wrap, edge_value=edge_value)
            sleep(1/fps)

    run_automaton._worker = run_threaded()
