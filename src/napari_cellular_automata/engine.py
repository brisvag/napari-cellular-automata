import numpy as np
import einops
from scipy.ndimage import convolve


def count_neighbours(world, nstates, wrap=True, edge_value=0):
    kernel = np.full((3,) * world.ndim, 1, dtype=np.uint8)
    kernel[(1,) * world.ndim] = 0
    counts = []
    for i in range(nstates):
        # fill with 1 beyond the edges only in the "background" case
        if i == edge_value:
            cval = 1
        else:
            cval = 0

        counts.append(
            convolve(
                # only consider 1 state at a time
                np.where(world == i, 1, 0),
                kernel,
                mode='wrap' if wrap else 'constant',
                cval=cval,
            )
        )
    return einops.rearrange(counts, 'neighbours ... -> ... neighbours')


def count_neighbours_survival_birth(world, wrap=True, edge_value=0):
    kernel = np.full((3,) * world.ndim, 1, dtype=np.uint8)
    kernel[(1,) * world.ndim] = 0

    return convolve(
        # only consider 1 state at a time
        np.where(world != 0, 1, 0),
        kernel,
        mode='wrap' if wrap else 'constant',
        cval=edge_value or 1,
    )


def _step_world(world, rules, wrap=True, edge_value=0):
    neighbours = count_neighbours(world, len(rules), wrap=wrap, edge_value=edge_value)
    new_world = np.zeros_like(world, dtype=np.uint8)
    for state, state_info in enumerate(rules):
        this_state = world == state
        new_world[this_state] = state_info['default']
        for neighbour_states, new_state in state_info.get('transitions', {}).items():
            relevant_state_indices = [n != -1 for n in neighbour_states]
            relevant_neighbours = neighbours[..., relevant_state_indices]
            relevant_states = np.array(neighbour_states)[relevant_state_indices]
            match = this_state & np.all(relevant_neighbours == relevant_states, axis=-1)
            new_world[match] = new_state
    return new_world


def _step_world_survival_birth(world, rules, wrap=True, edge_value=0):
    neighbours = count_neighbours_survival_birth(world, wrap=wrap, edge_value=edge_value)
    survive = np.zeros_like(world, dtype=bool)
    for rng in rules['survival']:
        if isinstance(rng, int):
            survive |= neighbours == rng
        else:
            survive |= (rng[0] <= neighbours) & (neighbours <= rng[1])
    born = np.zeros_like(neighbours, dtype=bool)
    for rng in rules['birth']:
        if isinstance(rng, int):
            born |= neighbours == rng
        else:
            born |= (rng[0] <= neighbours) & (neighbours <= rng[1])

    new_world = np.zeros_like(world, dtype=np.uint8)
    alive_state = rules['states'] - 1

    alive = world == alive_state
    empty = world == 0
    decaying = ~alive & ~empty
    new_world[alive & survive] = alive_state
    new_world[decaying] = world[decaying] - 1
    new_world[empty & born] = alive_state

    return new_world


def step_world(world, automaton, wrap=True, edge_value=0):
    if automaton['mode'] == 'full':
        return _step_world(world, automaton['rules'], wrap=wrap, edge_value=edge_value)
    elif automaton['mode'] == 'survival-birth':
        return _step_world_survival_birth(world, automaton['rules'], wrap=wrap, edge_value=edge_value)


def get_colors(automaton):
    if automaton['mode'] == 'full':
        return {i: state['color'] for i, state in enumerate(automaton['rules'])},
    if automaton['mode'] == 'survival-birth':
        greens = np.linspace(0, 1, automaton['rules']['states'] - 1)
        reds = 1 - greens
        return {i: (r, g, 0, 1) for i, (r, g) in enumerate(zip(reds, greens))}


def get_state_descriptions(automaton):
    if automaton['mode'] == 'full':
        return {state['name']: state['color'] for i, state in enumerate(automaton['rules'])},
    if automaton['mode'] == 'survival-birth':
        return {'alive': 'green', 'dead': 'transparent', 'decay': 'rest'}


def init_world(shape, percentages):
    world = np.zeros(shape, dtype=np.uint8)
    for state, percent in enumerate(percentages):
        world[np.random.rand(*shape) < percent] = state
    return world


# def rules_from_survival_birth(rules_string):
#     def ranges_to_neighbour_counts(ranges, states):
#         transitions = []
#         for rng in ranges.split(','):
#             if '-' in rng:
#                 low, high = [int(i) for i in rng.split('-')]
#             else:
#                 low = high = int(rng)
#             for n in range(low, high + 1):
#                 neighbour_counts = [-1] * states
#                 transitions.append((-1, n) + (-1,) * states)
#         return transitions
#
#     survival, birth, states = rules_string.split('/')
#
#     states = int(states)
#     decay_states = states - 2
#
#     rules = []
#
#     rules.append(
#         {
#             'transitions': {tr: states - 1 for tr in ranges_to_neighbour_counts(survival, states)},
#             'default': 0,
#             'color': 'transparent',
#             'name': 'dead',
#         }
#     )
#
#     for state in range(1, states - 1):
#         ratio = state / (states - 1)
#         rules.append(
#             {
#                 'default': state - 1,
#                 'color': (1 - ratio, ratio, 0, 1),
#                 'name': f'decay{decay_states - state}',
#             }
#         )
#
#     rules.append(
#         {
#             'transitions': {tr: states - 1 for tr in ranges_to_neighbour_counts(survival, states)},
#             'default': states - 2,
#             'color': (0, 1, 0, 1),
#             'name': 'alive',
#         }
#     )
#     return rules
#

def rules_from_survival_birth(rules_string):
    def parse_ranges(ranges):
        ranges_ = []
        for rng in ranges.split(','):
            if '-' in rng:
                ranges_.append(tuple(int(i) for i in rng.split('-')))
            else:
                ranges_.append(int(rng))
        return ranges_

    survival, birth, states = rules_string.split('/')

    return {
        'survival': parse_ranges(survival),
        'birth': parse_ranges(birth),
        'states': int(states),
    }
