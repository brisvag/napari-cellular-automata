#!/usr/bin/env python3

from time import sleep

import numpy as np
import pyqtgraph as pg
from magicgui import magic_factory
from napari import Viewer
from napari.layers import Labels
from napari.qt import thread_worker
from napari.utils.color import ColorValue

from .automata import AUTOMATA
from .engine import (
    get_colors,
    get_state_descriptions,
    populate_world_with_state,
    step_world,
)


def _update_text_overlay(viewer):
    selected = viewer.layers.selection.active
    if selected is None:
        viewer.text_overlay.visible = False
        return

    auto = selected.metadata.get("automaton", None)
    if auto is None:
        viewer.text_overlay.visible = False
        return

    viewer.text_overlay.update(
        {
            "visible": True,
            "position": "top_left",
            "text": (", ").join(
                f"{name}: {desc}"
                for name, desc in get_state_descriptions(auto).items()
            ),
        }
    )


@magic_factory(
    call_button="Generate",
    automaton={"choices": AUTOMATA.keys()},
    size={"widget_type": "Slider", "min": 10, "max": 1000},
)
def initialize_world(
    viewer: Viewer, automaton: str, size: int = 100
) -> Labels:
    selection = viewer.layers.selection
    if _update_text_overlay not in selection.events.callbacks:
        selection.events.connect(lambda: _update_text_overlay(viewer))

    auto = AUTOMATA[automaton]

    shape = (size,) * auto["ndim"]
    world = np.zeros(shape, dtype=np.uint8)

    lb = Labels(
        world,
        name=automaton,
        color=get_colors(auto),
        opacity=1,
        metadata={"automaton": auto},
    )
    lb.brush_size = 1
    lb.bounding_box.visible = True
    lb.bounding_box.points = False
    lb.bounding_box.line_color = "grey"

    return lb


def _get_state_choices(wdg):
    p = getattr(wdg, "_parent", None)
    if p is None or p.layer.value is None:
        return []
    return list(get_colors(p.layer.value.metadata["automaton"]).keys())


def _init_populator(wdg):
    wdg.state._parent = wdg


@magic_factory(
    call_button="Populate",
    state={"widget_type": "ComboBox", "choices": _get_state_choices},
    blob_density={"widget_type": "Slider", "min": 0, "max": 100000},
    blob_size={"widget_type": "Slider", "min": 1, "max": 100},
    blob_number={"widget_type": "Slider", "min": 1, "max": 20},
    blob_spread={"widget_type": "Slider", "min": 1, "max": 100},
    widget_init=_init_populator,
)
def populate_world(
    layer: Labels,
    state: int,
    fill=False,
    blob_density: int = 500,
    blob_size: int = 5,
    blob_number: int = 3,
    blob_spread: int = 20,
):
    if fill:
        layer.data[:] = state
    else:
        populate_world_with_state(
            layer.data,
            state,
            blob_density,
            blob_size,
            blob_number,
            blob_spread,
        )
    layer.refresh()


def _init_runner(wdg):
    wdg._plot = pg.PlotWidget()
    wdg._plot.addLegend()
    wdg.native.layout().addWidget(wdg._plot)
    wdg.plot_states._parent = wdg


@magic_factory(
    call_button="Run/Pause",
    fps={"widget_type": "Slider", "min": 1, "max": 120},
    widget_init=_init_runner,
    plot_states={"widget_type": "Select", "choices": _get_state_choices},
)
def run_automaton(
    viewer: Viewer,
    layer: Labels,
    fps=60,
    wrap=True,
    edge_value=0,
    plot_states=(),
):
    if getattr(run_automaton, "_worker", None) is not None:
        run_automaton._worker.quit()
        run_automaton._worker = None
        return

    selection = viewer.layers.selection
    if _update_text_overlay not in selection.events.callbacks:
        selection.events.connect(lambda: _update_text_overlay(viewer))

    # need to get it in the outer scope or magic_factory gets confused
    plot_widget = run_automaton._plot

    def set_data(data):
        layer.data = data
        stats = layer.metadata.setdefault("stats", {})
        plot_widget.clear()
        uniq, counts = np.unique(data, return_counts=True)
        for state, count in zip(uniq, counts):
            stats.setdefault(state, []).append(count)

        colors = get_colors(layer.metadata["automaton"])
        for state in plot_states:
            plot_widget.plot(
                stats[state],
                name=state,
                pen={"color": ColorValue.validate(colors[state]) * 255},
            )
        plot_widget.autoRange()

    rules = layer.metadata["automaton"]

    @thread_worker(connect={"yielded": set_data})
    def run_threaded():
        while True:
            yield step_world(
                layer.data, rules, wrap=wrap, edge_value=edge_value
            )
            sleep(1 / fps)

    run_automaton._worker = run_threaded()
