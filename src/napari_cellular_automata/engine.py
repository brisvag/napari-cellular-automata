import einops
import numpy as np
from scipy.ndimage import convolve


def _count_neighbours(world, nstates, wrap=True, edge_value=0):
    kernel = np.full((3,) * world.ndim, 1, dtype=np.uint8)
    kernel[(1,) * world.ndim] = 0
    counts = []
    for i in range(nstates):
        # fill with 1 beyond the edges only in the "background" case
        cval = 1 if i == edge_value else 0

        counts.append(
            convolve(
                # only consider 1 state at a time
                np.where(world == i, 1, 0),
                kernel,
                mode="wrap" if wrap else "constant",
                cval=cval,
            )
        )
    return einops.rearrange(counts, "neighbours ... -> ... neighbours")


def _count_neighbours_survival_birth(world, wrap=True, edge_value=0):
    kernel = np.full((3,) * world.ndim, 1, dtype=np.uint8)
    kernel[(1,) * world.ndim] = 0

    return convolve(
        # only consider 1 state at a time
        np.where(world != 0, 1, 0),
        kernel,
        mode="wrap" if wrap else "constant",
        # either 1 or 0 for edge value, since type of neighbor does not matter
        cval=1 if edge_value else 0,
    )


def _step_world(world, rules, wrap=True, edge_value=0):
    neighbours = _count_neighbours(
        world, len(rules), wrap=wrap, edge_value=edge_value
    )
    new_world = np.zeros_like(world, dtype=np.uint8)
    for state, state_info in enumerate(rules):
        this_state = world == state
        new_world[this_state] = state_info["default"]
        for neighbour_states, new_state in state_info.get(
            "transitions", {}
        ).items():
            relevant_state_indices = [n != -1 for n in neighbour_states]
            relevant_neighbours = neighbours[..., relevant_state_indices]
            relevant_states = np.array(neighbour_states)[
                relevant_state_indices
            ]
            match = this_state & np.all(
                relevant_neighbours == relevant_states, axis=-1
            )
            new_world[match] = new_state
    return new_world


def _step_world_survival_birth(world, rules, wrap=True, edge_value=0):
    neighbours = _count_neighbours_survival_birth(
        world, wrap=wrap, edge_value=edge_value
    )
    survive = np.zeros_like(world, dtype=bool)
    for rng in rules["survival"]:
        if isinstance(rng, int):
            survive |= neighbours == rng
        else:
            survive |= (rng[0] <= neighbours) & (neighbours <= rng[1])
    born = np.zeros_like(neighbours, dtype=bool)
    for rng in rules["birth"]:
        if isinstance(rng, int):
            born |= neighbours == rng
        else:
            born |= (rng[0] <= neighbours) & (neighbours <= rng[1])

    new_world = np.zeros_like(world, dtype=np.uint8)
    alive_state = rules["states"] - 1

    alive = world == alive_state
    empty = world == 0
    decaying = ~empty & ~survive
    new_world[alive & survive] = alive_state
    new_world[decaying] = world[decaying] - 1
    new_world[empty & born] = alive_state

    return new_world


def step_world(world, automaton, wrap=True, edge_value=0):
    if automaton["mode"] == "full":
        return _step_world(
            world, automaton["rules"], wrap=wrap, edge_value=edge_value
        )
    elif automaton["mode"] == "survival-birth":
        return _step_world_survival_birth(
            world, automaton["rules"], wrap=wrap, edge_value=edge_value
        )


def get_colors(automaton):
    if automaton["mode"] == "full":
        return {
            i: state["color"] for i, state in enumerate(automaton["rules"])
        }
    if automaton["mode"] == "survival-birth":
        greens = np.linspace(0, 1, automaton["rules"]["states"] - 1)
        reds = 1 - greens
        return {
            0: (0, 0, 0, 0),
            **{
                i + 1: (r, g, 0, 1)
                for i, (r, g) in enumerate(zip(reds, greens))
            },
        }


def get_state_descriptions(automaton):
    if automaton["mode"] == "full":
        return {
            state["name"]: state["color"]
            for i, state in enumerate(automaton["rules"])
        }
    if automaton["mode"] == "survival-birth":
        states = automaton["rules"]["states"]
        decays = {f"decay{i}": "r/g" for i in range(states - 2, 0, -1)}
        return {"dead": "transparent", **decays, "alive": "green"}


def populate_world_with_state(
    world, state, blob_density=500, blob_size=5, blob_number=3, blob_spread=20
):
    blob_centers = np.random.normal(
        np.array(world.shape) / 2, blob_spread, (blob_number, world.ndim)
    )
    all_samples = []
    for center in blob_centers:
        samples = np.random.normal(
            center, blob_size, (blob_density, world.ndim)
        )
        all_samples.append(samples)
    bins = [np.arange(s + 1) - 0.5 for s in world.shape]
    binned = np.histogramdd(np.concatenate(all_samples), bins=bins)[0]
    world[binned != 0] = state


def rules_from_survival_birth(rules_string):
    def parse_ranges(ranges):
        ranges_ = []
        for rng in ranges.split(","):
            if "-" in rng:
                ranges_.append(tuple(int(i) for i in rng.split("-")))
            else:
                ranges_.append(int(rng))
        return ranges_

    survival, birth, states = rules_string.split("/")

    return {
        "survival": parse_ranges(survival),
        "birth": parse_ranges(birth),
        "states": int(states),
    }
