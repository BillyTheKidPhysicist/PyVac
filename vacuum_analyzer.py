"""
Contains methods and objects for computing vacuum system performance. A model of a vacuum is created component by
component, then the system of equations of conductance and pump speed are converted to a matrix and solved. 

The model is simple and currently only allows a linear vacuum system composed of pumping regions, constants pressure 
regions, apertures, and long tubes, with pressure values computed at each pump.
"""
import warnings
from math import sqrt
from typing import Union, Iterable

import numpy as np

from constants import ROOM_TEMPERATURE, MOLAR_MASSES

RealNum = Union[float, int]
N2_mass = MOLAR_MASSES['N2']
long_tube_LD_ratio = 5  # A long tube has a length ot diameter ration of about 5. Less than this and the approximation


# is dubious


def long_tube_conductance(D, L, T=ROOM_TEMPERATURE, m=N2_mass) -> float:
    """Return the conductance of a long tube (L>>D)"""
    geometric_factor = D ** 3 / L
    gas_factor = 3.81 * sqrt(T / m)
    return gas_factor * geometric_factor


def aperture_conductance(D, T=ROOM_TEMPERATURE, m=N2_mass):
    """Return the conductance of an aperture"""
    return 2.9 * sqrt(T / m) * D ** 2


def check_long_tube_params(L, D):
    """Check that the parameters for a long tube are consistent with the approximation. Raise a warning if not"""
    if L < 0 or D < 0:
        raise ValueError("L and D must both be greater than 0")
    if L < D * 5:
        warnings.warn("Tube length must be greater than about 5 times the tube diameter for valid results")


def check_chamber_parameters(S, Q, P):
    """Check that the parameters of a chamber are consistent."""
    if S < 0 or Q < 0 or P < 0:
        raise ValueError("S,Q, and P must all be >=0")
    if P != 0 and (S > 0 or Q > 0):  # initial pressure is a constaint, and would clash with Q and S
        raise ValueError("If the pressure is specified as non zero, the pumping speed and gas load must be zero")


class Tube:
    """A class to represent a long tube"""

    def __init__(self, L: RealNum, D: RealNum, name: str):
        assert L > 0.0 and D > 0.0
        self.name = name  # user specified name
        self.L = L  # length, m
        self.D = D  # inside diameter, m

    def C(self) -> float:
        """Return the conductance of the tube"""
        return long_tube_conductance(self.D, self.L)


class Chamber:
    """A class to represent a chamber region. A constant pressure can be specified, or a gas load and/or pumping
    speed."""

    def __init__(self, S: RealNum, Q: RealNum, P: RealNum, name: str):
        self.name = name  # user specified name
        self._S = S  # pump speed, L/s
        self._Q = Q  # gas load, Torr L/s
        self._P = P  # pressure, Torr

    def S(self):
        """Return the pump speed of the chamber"""
        return self._S

    def Q(self):
        """Return the user specified gas load in the chamber"""
        return self._Q

    def P(self):
        """Return the user specified constant pressure in the chamber"""
        return self._P


def total_conductance(components) -> float:
    """Return the total conductance of a provided components assumed to be in series"""
    return 1 / sum(1 / component.C() for component in components)


def node_index(node, nodes: list) -> int:
    """Return the index of the node in nodes. Assumed to be a linear sequence of nodes"""
    indices = [i for i, _node in enumerate(nodes) if _node is node]
    assert len(indices) == 1
    return indices[0]


def branches_from_node(branch_node, nodes):
    """Return the branches from the node in nodes. Assumed to be a linear sequence of nodes"""
    index = node_index(branch_node, nodes)
    nodes_forward = [nodes[i] for i in range(index + 1, len(nodes))]
    nodes_backward = [nodes[i] for i in range(index - 1, -1, -1)]
    return nodes_backward, nodes_forward


def terminate_branch_at_chamber(nodes: list):
    """For a sequences of nodes (vacuum components), the sequence of nodes from the beginning to the next vacuum
    chamber"""
    if len(nodes) == 0:
        return []
    branch = []
    for node in nodes:
        branch.append(node)
        if type(node) is Chamber:
            break
    return branch


def branches_to_neighbor_chambers(branch_node, nodes: list):
    """Return the sequences of nodes from the branch_node that end in chambers"""
    nodes_backward, nodes_forward = branches_from_node(branch_node, nodes)
    branch_forward = terminate_branch_at_chamber(nodes_forward)
    branch_backward = terminate_branch_at_chamber(nodes_backward)
    return branch_backward, branch_forward


def make_Q_vec(chambers):
    """Return the gas load vector"""
    Q_vec = np.array([chamber.Q() for chamber in chambers])
    for i, chamber_a in enumerate(chambers):
        branches = branches_to_neighbor_chambers(chamber_a, chambers)
        for branch in branches:
            if len(branch) != 0:
                tubes, chamber_b = branch[:-1], branch[-1]
                C_total = total_conductance(tubes)
                if chamber_b.P() is not None:
                    Q_vec[i] += C_total * chamber_b.P()
    return Q_vec


def make_C_matrix(chambers):
    """Return the conductance matrix"""
    dim = len(chambers)
    C_matrix = np.zeros((dim, dim))
    for idx_a, chamber_a in enumerate(chambers):
        C_matrix[idx_a, idx_a] += chamber_a.S()
        branches = branches_to_neighbor_chambers(chamber_a, chambers)
        for branch in branches:
            assert len(branch) != 1  # either no branch, or at least one tube, then a vacuum chamber
            if len(branch) != 0:
                tubes, chamber_b = branch[:-1], branch[-1]
                C_total = total_conductance(tubes)
                C_matrix[idx_a, idx_a] += C_total
                if chamber_b.P() is None:
                    idx_b = chambers.index(chamber_b)
                    C_matrix[idx_a, idx_b] += -C_total
    return C_matrix


def solve_system(vac_sys):
    """
    Solve for pressure in the vacuum system, return a dictionary that maps chambers to their pressure

    :param vac_sys:
    :return:
    """
    unconstrained_chambers = []
    for chamber in vac_sys.chambers():
        if chamber.P() is None:
            unconstrained_chambers.append(chamber)

    Q = make_Q_vec(unconstrained_chambers)
    C = make_C_matrix(unconstrained_chambers)
    _P = np.linalg.inv(C) @ Q

    P = {}
    for chamber in vac_sys.chambers():
        if chamber in unconstrained_chambers:
            P[chamber] = _P[unconstrained_chambers.index(chamber)]
        else:
            P[chamber] = chamber.P()
    return P


Component = Union[Tube, Chamber]


class VacuumSystem:
    """A model of the vacuum system. The user adds components one after another"""

    def __init__(self):
        self.components: list[Component] = []
        self.graph = {}

    def add_long_tube(self, L: float, D: float, name: str = 'unassigned'):
        """
        Add a long tube to the vacuum system. Length must be sufficiently longer than the diameter for accuracy
        
        :param L: Length of the tube, m
        :param D: Inside diameter of the tube, m
        :param name: User specified name
        :return: None
        """
        check_long_tube_params(L, D)
        component = Tube(L, D, name)
        self.components.append(component)

    def add_chamber(self, S: float = 0.0, Q: float = 0.0, P: float = None, name: str = 'unassigned'):
        """
        Add a chamber to the vacuum system.

        If the pressure is specified, the gas load and pumping speed must be zero. The specified  pressure is assumed
        to be a constant of the chamber and would implicitly account for all gas loads and pumping speeds.

        :param S: Pumping speed, L/s
        :param Q: Gas load inside the chamber, not from other connected chamber, Torr L/s
        :param P: The pressure of the chamber, Torr
        :param name: User specified name
        :return: None
        """
        check_chamber_parameters(S, Q, P)
        component = Chamber(S, Q, P, name)
        self.components.append(component)

    def chambers(self):
        """List of chamber in the system"""
        return [comp for comp in self if type(comp) is Chamber]

    def __iter__(self) -> Iterable[Component]:
        return iter(self.components)

    def __len__(self) -> int:
        return len(self.components)

    def __getitem__(self, index: int) -> Component:
        return self.components[index]
