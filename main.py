# Fully differentiable 2D cellular potts model powered by JAX

from dataclasses import dataclass
from functools import partial

import jax
from jax import jit
from jax import numpy as np
import numpy as onp

from matplotlib import pyplot as plt

jit = partial(jit, backend="cpu")


@dataclass
class System:
    grid: np.ndarray
    cell_types: np.ndarray


def setup(
    grid_size: tuple[int, int] = (100, 100),
    start_pos: tuple[int, int] = (20, 20),
    end_pos: tuple[int, int] = (80, 80),
    volume: float = 25,
) -> System:
    """Setup the simulation

    Set up a simulation with 50% of the grid between `start_pos` and `end_pos` filled
    with cells of type 1 and the other 50% with cells of type 2.
    """
    cell_size = int(onp.sqrt(volume))
    width = end_pos[0] - start_pos[0]
    height = end_pos[1] - start_pos[1]
    n_cells = int(width * height / cell_size**2)

    grid = onp.zeros(grid_size, dtype=onp.int32)

    cell_index = 1
    for x in range(start_pos[0], end_pos[0], cell_size):
        for y in range(start_pos[1], end_pos[1], cell_size):
            grid[x : x + cell_size, y : y + cell_size] = cell_index
            cell_index += 1
    cell_types = onp.zeros(n_cells + 1, dtype=onp.int32)
    cell_types[1:] = onp.random.choice([1, 2], size=n_cells)

    return System(np.array(grid), np.array(cell_types))


def plot_system(system: System):
    """Plot the current system state"""
    plt.imshow(np.asarray(system.cell_types[system.grid]))
    plt.show()


@jit
def cell_interaction_energy(
    cell_id: int,
    cell_type: int,
    grid: np.ndarray,
    cell_types: np.ndarray,
    J: np.ndarray,
    V: float,
    lambda_v: float,
    x: int,
    y: int,
) -> float:
    energy = 0
    energy += np.where(
        np.logical_and(x > 0, cell_id != grid[x - 1, y]),
        J[cell_type, cell_types[x - 1, y]],
        0,
    )
    energy += np.where(
        np.logical_and(x < grid.shape[0] - 1, cell_id != grid[x + 1, y]),
        J[cell_type, cell_types[x + 1, y]],
        0,
    )
    energy += np.where(
        np.logical_and(y > 0, cell_id != grid[x, y - 1]),
        J[cell_type, cell_types[x, y - 1]],
        0,
    )
    energy += np.where(
        np.logical_and(y < grid.shape[1] - 1, cell_id != grid[x, y + 1]),
        J[cell_type, cell_types[x, y + 1]],
        0,
    )
    return energy


@partial(jit, static_argnums=(5,))
def hamiltonian(
    grid: np.ndarray,
    cell_types: np.ndarray,
    J: np.ndarray,
    V: float,
    lambda_v: float,
    n_cells: int,
) -> float:
    """Compute the energy of the system

    Args:
        system: The current system state
        J: The interaction matrix between cell types
        V: The target volume of the cells
        lambda_v: The volume constraint strength
        n_cells: The number of cells in the system
    """
    cell_types = cell_types[grid]
    energy = 0

    # Interaction energy
    @jit
    def energy_fn(x, y):
        return cell_interaction_energy(
            grid[x, y], cell_types[x, y], grid, cell_types, J, V, lambda_v, x, y
        )

    xx, yy = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]))
    energy = np.sum(jax.vmap(energy_fn)(xx.flatten(), yy.flatten()))

    # Volume constraint energy
    energy += lambda_v * np.sum(
        (np.bincount(grid.flatten(), length=n_cells + 1)[1:] - V) ** 2
    )

    return energy


@jit
def delta_energy(
    grid: np.ndarray,
    cell_types: np.ndarray,
    flip_x: int,
    flip_y: int,
    new_cell_id: int,
    J: np.ndarray,
    V: float,
    lambda_v: float,
) -> float:
    """Compute the energy difference of the system after flipping a cell

    Args:
        grid: The current system state
        cell_types: The cell types of the cells in the system
        flip_x: The x coordinate of the cell to flip
        flip_y: The y coordinate of the cell to flip
        new_cell_id: The cell id of the cell to flip
        J: The interaction matrix between cell types
        V: The target volume of the cells
        lambda_v: The volume constraint strength
    """
    old_cell_id = grid[flip_x, flip_y]
    old_volume_old_cell = np.sum(np.where(grid == old_cell_id, 1, 0))
    new_volume_old_cell = old_volume_old_cell - 1
    old_volume_new_cell = np.sum(np.where(grid == new_cell_id, 1, 0))
    new_volume_new_cell = old_volume_new_cell + 1

    d_volume_energy_old_cell = np.where(
        old_cell_id == 0,
        0,
        lambda_v * ((new_volume_old_cell - V) ** 2 - (old_volume_old_cell - V) ** 2),
    )
    d_volume_energy_new_cell = np.where(
        new_cell_id == 0,
        0,
        lambda_v * ((new_volume_new_cell - V) ** 2 - (old_volume_new_cell - V) ** 2),
    )

    old_type = cell_types[grid[flip_x, flip_y]]
    new_type = cell_types[new_cell_id]

    old_neighbour_energy = cell_interaction_energy(
        old_cell_id, old_type, grid, cell_types[grid], J, V, lambda_v, flip_x, flip_y
    )
    new_neighbour_energy = cell_interaction_energy(
        new_cell_id, new_type, grid, cell_types[grid], J, V, lambda_v, flip_x, flip_y
    )

    return (
        d_volume_energy_old_cell
        + d_volume_energy_new_cell
        + 2 * new_neighbour_energy
        - 2 * old_neighbour_energy
    )


@jit
def propose_flip(
    key, grid: np.ndarray, cell_types: np.ndarray
) -> tuple[np.ndarray, int, int, int]:
    # select a random cell
    x = -1
    y = -1
    new_cell_id = -1

    def cond(args):
        key, x, y, cell_id, neighbours = args
        return np.all(neighbours == cell_id)

    def body(args):
        key, x, y, cell_id, neighbours = args
        key, subkey = jax.random.split(key)
        x = jax.random.randint(subkey, (1,), 0, grid.shape[0])[0]
        key, subkey = jax.random.split(key)
        y = jax.random.randint(subkey, (1,), 0, grid.shape[1])[0]

        cell_id = grid[x, y]
        # get surrounding cell types
        neighbours = np.hstack(
            [
                grid[np.maximum(0, x - 1), y],
                grid[np.minimum(x + 1, grid.shape[0] - 1), y],
                grid[x, np.maximum(0, y - 1)],
                grid[x, np.minimum(y + 1, grid.shape[1] - 1)],
            ]
        )
        return key, x, y, cell_id, neighbours

    key, x, y, cell_id, neighbours = jax.lax.while_loop(
        cond, body, (key, x, y, -1, np.array([-1, -1, -1, -1]))
    )

    u_neighbours = np.unique(neighbours, size=4, fill_value=cell_id)
    p = np.where(u_neighbours != cell_id, 1, 0)
    p /= np.sum(p)

    key, subkey = jax.random.split(key)
    new_cell_id = jax.random.choice(subkey, u_neighbours, p=p)

    return key, x, y, new_cell_id


@jit
def accept_flip(grid: np.ndarray, flip_x, flip_y, new_cell_id) -> np.ndarray:
    return grid.at[flip_x, flip_y].set(new_cell_id)


@jit
def mc_sweep(
    key,
    grid: np.ndarray,
    energy: float,
    cell_types: np.ndarray,
    J: np.ndarray,
    V: float,
    lambda_v: float,
    TEMPERATURE: float,
):
    def negative_de(key, x, y, new_cell_id, grid, de):
        return key, accept_flip(grid, x, y, new_cell_id), de

    def positive_de(key, x, y, new_cell_id, grid, de):
        key, subkey = jax.random.split(key)
        randn = jax.random.uniform(subkey) - np.exp(-de / TEMPERATURE)
        return key, np.where(
            randn < 0,
            accept_flip(grid, x, y, new_cell_id),
            grid,
        ), np.where(randn < 0, de, 0)

    def loop_body(_, args):
        key, grid, energy = args
        key, x, y, new_cell_id = propose_flip(key, grid, cell_types)
        de = delta_energy(grid, cell_types, x, y, new_cell_id, J, V, lambda_v)

        key, grid, de = jax.lax.cond(
            de < 0, negative_de, positive_de, key, x, y, new_cell_id, grid, de
        )

        return key, grid, energy + de

    key, grid, energy = jax.lax.fori_loop(
        0, grid.shape[0] * grid.shape[1], loop_body, (key, grid, energy)
    )

    # for i in range(grid.shape[0] * grid.shape[1]):
    # if de < 0:
    #     grid = np.where(de < 0, accept_flip(grid, x, y, new_cell_id), grid)
    #     energy += np.where(de < 0, de, 0)
    # else:
    #     key, subkey = jax.random.split(key)
    #     if jax.random.uniform(subkey) < np.exp(-de / TEMPERATURE):
    #         grid = accept_flip(grid, x, y, new_cell_id)
    #         energy += de
    return grid, energy


J = np.array(
    [
        [0, 16, 16],
        [16, 2, 11],
        [16, 11, 16],
    ]
)

V = 25
lambda_v = 2
TEMPERATURE = 10

key = jax.random.PRNGKey(123)


system = setup()
n_cells = int(np.max(system.grid))
energy = hamiltonian(system.grid, system.cell_types, J, V, lambda_v, n_cells)
start_energy = energy

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
# img = ax.imshow(np.asarray(system.cell_types[system.grid]))
img = ax.imshow(np.asarray(system.grid))

for i in range(10):
    system.grid, energy = mc_sweep(
        key, system.grid, energy, system.cell_types, J, V, lambda_v, TEMPERATURE
    )
    img.set_data(np.asarray(system.grid))
    fig.canvas.draw()
    fig.canvas.flush_events()
    print(f"Energy: {energy}")
    onp.savetxt(f"grid_{i}.txt", np.asarray(system.grid), fmt="%d")


print(f"Start energy: {start_energy}")
print(f"Energy from de: {energy}")
print(
    f"Energy from hamiltonian: {hamiltonian(system.grid, system.cell_types, J, V, lambda_v, n_cells)}"
)

# system = setup()
# x = 20
# y = 20
# new_cell_id = 0
# energy_start = hamiltonian(system.grid, system.cell_types, J, V, lambda_v, n_cells)
# de = delta_energy(system.grid, system.cell_types, x, y, new_cell_id, J, V, lambda_v)
# system.grid = accept_flip(system.grid, x, y, new_cell_id)
# energy_end = hamiltonian(system.grid, system.cell_types, J, V, lambda_v, n_cells)
# print(f"denergy: {de}")
# print(f"denergy hamiltonian: {energy_end - energy_start}")
