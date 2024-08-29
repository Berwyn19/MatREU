from typing import Optional, Tuple

import numpy as np
from matscipy.neighbours import neighbour_list

def get_neighborhood_layered(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
    true_self_interaction=False,
    atomic_numbers=None,
):
    # AR: If this is a 2D material, follow these steps:
    # 1) Check if material is 2D (ASE) | Assign each atom a layer ID
    # 2) Construct neighbor_list dict | Only include interlayer interactions

    from ase.geometry.dimensionality import analyze_dimensionality
    from ase.atoms import Atoms
    import itertools

    # We want to keep this line, but it makes to output logs close to unreadable
    # It's still definetly a greppable output
    # logging.warning(
    #    f"!!get_neighborhood_layered only works with bilayer datasets at present!!"
    # )
    nums = atomic_numbers
    if nums is None:
        nums = np.zeros(positions.shape[0])
    atoms = Atoms(
        positions=positions,
        numbers=nums,
        cell=cell,
        pbc=pbc,
    )

    intervals = analyze_dimensionality(atoms, method="RDA")

    m = intervals[0]

    tmp_at_num = np.array(m.components, dtype=np.int32)  # int32 for matscipy issues
    # tmp_at_num = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int32)
    max_layer = np.max(tmp_at_num)

    # tmp_at_num = np.array(
    #    np.hstack((0 * np.ones(81), 1 * np.ones(159 - 81))), dtype=np.int32
    # )

    # print(tmp_at_num)
    # max_layer = 1
    # print(tmp_at_num, range(max_layer))

    cutoff_dict = {}
    for comb in itertools.combinations(range(max_layer + 1), 2):
        # print(comb)
        cutoff_dict.update({comb: cutoff})
    if pbc is None:
        pbc = (False, False, False)

    if cell is None or cell.any() == np.zeros((3, 3)).any():
        cell = np.identity(3, dtype=float)

    assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
    assert cell.shape == (3, 3)

    pbc_x = pbc[0]
    pbc_y = pbc[1]
    pbc_z = pbc[2]
    identity = np.identity(3, dtype=float)
    max_positions = np.max(np.absolute(positions)) + 1
    # Extend cell in non-periodic directions
    # For models with more than 5 layers, the multiplicative constant needs to be increased.
    if not pbc_x:
        cell[:, 0] = max_positions * 5 * cutoff * identity[:, 0]
    if not pbc_y:
        cell[:, 1] = max_positions * 5 * cutoff * identity[:, 1]
    if not pbc_z:
        cell[:, 2] = max_positions * 5 * cutoff * identity[:, 2]

    sender, receiver, unit_shifts = neighbour_list(
        quantities="ijS",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff_dict,
        numbers=tmp_at_num,
        # self_interaction=True,  # we want edges from atom to itself in different periodic images
        # use_scaled_positions=False,  # positions are not scaled positions
    )

    if not true_self_interaction:
        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = sender == receiver
        true_self_edge &= np.all(unit_shifts == 0, axis=1)
        keep_edge = ~true_self_edge

        # Note: after eliminating self-edges, it can be that no edges remain in this system
        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]

    # Build output
    edge_index = np.stack((sender, receiver))  # [2, n_edges]

    # From the docs: With the shift vector S, the distances D between atoms can be computed from
    # D = positions[j]-positions[i]+S.dot(cell)
    shifts = np.dot(unit_shifts, cell)  # [n_edges, 3]

    return edge_index, shifts, unit_shifts

def get_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
    true_self_interaction=False,
) -> Tuple[np.ndarray, np.ndarray]:
    if pbc is None:
        pbc = (False, False, False)

    if cell is None or cell.any() == np.zeros((3, 3)).any():
        cell = np.identity(3, dtype=float)

    assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
    assert cell.shape == (3, 3)

    pbc_x = pbc[0]
    pbc_y = pbc[1]
    pbc_z = pbc[2]
    identity = np.identity(3, dtype=float)
    max_positions = np.max(np.absolute(positions)) + 1
    # Extend cell in non-periodic directions
    # For models with more than 5 layers, the multiplicative constant needs to be increased.
    temp_cell = np.copy(cell)
    if not pbc_x:
        temp_cell[0, :] = max_positions * 5 * cutoff * identity[0, :]
    if not pbc_y:
        temp_cell[1, :] = max_positions * 5 * cutoff * identity[1, :]
    if not pbc_z:
        temp_cell[2, :] = max_positions * 5 * cutoff * identity[2, :]

    sender, receiver, unit_shifts = neighbour_list(
        quantities="ijS",
        pbc=pbc,
        cell=temp_cell,
        positions=positions,
        cutoff=cutoff,
        # self_interaction=True,  # we want edges from atom to itself in different periodic images
        # use_scaled_positions=False,  # positions are not scaled positions
    )

    if not true_self_interaction:
        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = sender == receiver
        true_self_edge &= np.all(unit_shifts == 0, axis=1)
        keep_edge = ~true_self_edge

        # Note: after eliminating self-edges, it can be that no edges remain in this system
        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]

    # Build output
    edge_index = np.stack((sender, receiver))  # [2, n_edges]

    # From the docs: With the shift vector S, the distances D between atoms can be computed from
    # D = positions[j]-positions[i]+S.dot(cell)
    shifts = np.dot(unit_shifts, cell)  # [n_edges, 3]

    return edge_index, shifts, unit_shifts
