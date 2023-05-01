from napari_cellular_automata.widgets import initialize_world


def test_widget(make_napari_viewer):
    viewer = make_napari_viewer()

    # create our widget, passing in the viewer
    my_widget = initialize_world()
    viewer.window.add_dock_widget(my_widget)
    my_widget()

    assert len(viewer.layers) == 1
    assert "automaton" in viewer.layers[0].metadata
