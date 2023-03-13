"""
Contains methods and objects for computing vacuum system performance. A model of a vacuum is created component by
component, then the system of equations of conductance and pump speed are converted to a matrix and solved. Currently
only supports linear vacuum systems, but a circular vacuum system with gradual bender is basically a linear vacuum
system
"""
import copy
import warnings
from math import pi, sqrt
from typing import Optional
from typing import Union, Iterable, Any

import matplotlib.pyplot as plt
import numpy as np

from constants import ROOM_TEMPERATURE, BOLTZMANN_CONSTANT,rate_coefficients
from utilities import is_ascending

RealNum = Union[float, int]

R = 8.3145


def tube_conductance(m_Daltons, inside_diam, L, T=ROOM_TEMPERATURE) -> float:
    geometric_factor = inside_diam ** 3 / L
    gas_factor = 3.81 * sqrt(T / m_Daltons)
    return gas_factor * geometric_factor


def aperture_conductance(m_Daltons, diameter, T=ROOM_TEMPERATURE):
    return 2.9 * sqrt(T / m_Daltons) * diameter ** 2


def split_length_into_equal_offset_array(L: RealNum, num_points: int) -> np.ndarray:
    num_splits = num_points + 1
    start = L / num_splits
    stop = L - start
    return np.linspace(start, stop, num_points)


def append_or_extend_to_list(item: Union[Any, list], item_list: list) -> None:
    if has_len(item):
        item_list.extend(item)
    else:
        item_list.append(item)


def has_len(obj: Any) -> bool:
    return hasattr(obj, '__len__')


def nan_array(num: int) -> np.ndarray:
    return np.ones(num) * np.nan


class _Component:
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        string = '--Component information-- \n'
        string += 'component type: ' + type(self).__name__ + '\n'
        string += 'name: ' + self.name + '\n'

        for key, value in self.__dict__.items():
            if key != 'name':
                string += key + ': ' + str(value) + '\n'
        return string


class Tube(_Component):
    def __init__(self, L: RealNum, inside_diam: RealNum, q: RealNum = 0.0, num_profile_points: int = 30,
                 name: str = 'unassigned'):
        assert L > 0.0 and inside_diam > 0.0
        if not L >= 3 * inside_diam:
            warnings.warn('Tube length should be several times longer than diam for more accurate results')
        super().__init__(name)
        self.P_x_vals = split_length_into_equal_offset_array(L, num_profile_points)
        self.P = nan_array(num_profile_points)
        self.L = L
        self.inside_diam = inside_diam
        self.Q = q * self.inside_diam * pi * self.L

    def C(self, m_Daltons, T=ROOM_TEMPERATURE) -> float:
        return tube_conductance(m_Daltons, self.inside_diam, self.L, T=T)


class Chamber(_Component):
    def __init__(self, S: RealNum, Q: RealNum, P: Optional[RealNum], name: str = 'unassigned', input_tube: Tube = None):
        assert not (P is not None and Q > 0.0)
        assert S >= 0.0 and Q >= 0.0 and (P > 0.0 if P is not None else True)
        super().__init__(name)
        self.P = P
        self._S = S
        self.Q = Q
        self.input_tube = input_tube

    def S(self, m_Daltons=None, T=ROOM_TEMPERATURE) -> float:
        if self._S == 0.0:
            return 0.0
        else:
            C_throat = np.inf if self.input_tube is None else self.input_tube.C(m_Daltons, T=T)
            speed = 1 / (1 / self._S + 1 / C_throat)
            return speed


class Aperture(_Component):
    def __init__(self, diameter, name: str = 'unassigned'):
        super().__init__(name)
        self.diameter = diameter

    def C(self, m_Daltons, T=ROOM_TEMPERATURE) -> float:
        return aperture_conductance(m_Daltons, self.diameter, T=T)


Component = Union[Tube, Chamber, Aperture]


def node_index(node, nodes: list) -> int:
    indices = [i for i, _node in enumerate(nodes) if _node is node]
    assert len(indices) == 1
    return indices[0]


def get_branch(nodes: list, stop_cond):
    if len(nodes) == 0:
        return []
    branch = []
    for node in nodes:
        branch.append(node)
        if stop_cond(node):
            break
    return branch


def reverse(some_list):
    some_list_copy = some_list.copy()
    some_list_copy.reverse()
    return some_list_copy


def backward_and_forward_nodes(branch_node, nodes, is_circular) -> tuple[list[Component], list[Component]]:
    index = node_index(branch_node, nodes)
    nodes_forward_to_end = [nodes[i] for i in range(index + 1, len(nodes))]
    nodes_backward_to_end = [nodes[i] for i in range(index - 1, -1, -1)]
    if is_circular:
        nodes_forward = [*nodes_forward_to_end, *reverse(nodes_backward_to_end)]
        nodes_backward = [*nodes_backward_to_end, *reverse(nodes_forward_to_end)]
    else:
        nodes_forward, nodes_backward = nodes_forward_to_end, nodes_backward_to_end
    return nodes_backward, nodes_forward


def get_branches_from_node(branch_node, nodes: list, stop_cond, is_circular) -> tuple[list, list]:
    nodes_backward, nodes_forward = backward_and_forward_nodes(branch_node, nodes, is_circular)
    branch_forward = get_branch(nodes_forward, stop_cond)
    branch_backward = get_branch(nodes_backward, stop_cond)
    return branch_backward, branch_forward


class VacuumSystem:
    def __init__(self, is_circular=False, gas_mass_Daltons=28):
        self.components: list[Component] = []
        self.is_circular = is_circular
        self.gas_mass = gas_mass_Daltons
        self.P_mean = None

    def add_tube(self, L: float, inside_diam: float, q: float = 0.0, name: str = 'unassigned'):
        component = Tube(L, inside_diam, q=q, name=name)
        self.components.append(component)

    def add_chamber(self, S: float = 0.0, Q: float = 0.0, name: str = 'unassigned', P: float = None,
                    input_tube: Tube = None):
        component = Chamber(S, Q, P, name=name, input_tube=input_tube)
        self.components.append(component)

    def add_aperture(self, diameter: float, name: str = 'unassigned'):
        component = Aperture(diameter, name=name)
        self.components.append(component)

    def num_components(self) -> int:
        return len(self.components)

    def chambers(self) -> tuple[Chamber]:
        return tuple([component for component in self if type(component) is Chamber])

    def __iter__(self) -> Iterable[Component]:
        return iter(self.components)

    def __len__(self) -> int:
        return len(self.components)

    def __getitem__(self, index: int) -> Component:
        return self.components[index]

    def branches(self, chamber: Chamber) -> tuple[list[Component], list[Component]]:
        assert type(chamber) is Chamber

        def stop_condition(node):
            return type(node) is Chamber

        branch_1, branch_2 = get_branches_from_node(chamber, self.components, stop_condition, self.is_circular)
        return branch_1, branch_2


class SolverVacuumSystem(VacuumSystem):
    def __init__(self, vacuum_system: VacuumSystem):
        super().__init__(is_circular=vacuum_system.is_circular, gas_mass_Daltons=vacuum_system.gas_mass)
        self.components, self.component_map = self.solver_components_and_map(vacuum_system)
        self.matrix_index: dict[Component, int] = self.solver_matrix_index_dict()

    def chambers_unsolved_pressure(self):
        return [chamb for chamb in self.chambers() if chamb.P is None]

    def solver_matrix_index_dict(self) -> dict[Component, int]:
        indices = range(len(self.chambers_unsolved_pressure()))
        matrix_index = dict(zip(self.chambers_unsolved_pressure(), indices))
        return matrix_index

    def split_tube(self, tube: Tube) -> list[Tube, Chamber]:
        num_profile_points = len(tube.P)
        L_split = tube.L / (num_profile_points + 1)
        Q_splits = tube.Q / num_profile_points
        component_initializers = [Tube, Chamber] * num_profile_points
        component_initializers.append(Tube)
        new_components = []
        for component_init in component_initializers:
            if component_init is Tube:
                new_components.append(component_init(L_split, tube.inside_diam, name=tube.name))
            elif component_init is Chamber:
                new_components.append(component_init(0.0, Q_splits, None))
            else:
                raise NotImplementedError

        return new_components

    def convert_component(self, component: Component) -> Union[list[Component], Component]:
        if type(component) in (Chamber, Aperture):
            solver_version = copy.copy(component)
        elif type(component) is Tube:
            if has_len(component.P):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    solver_version = self.split_tube(component)
            else:
                solver_version = copy.copy(component)
        else:
            raise NotImplementedError
        return solver_version

    def solver_components_and_map(self, vacuum_system: VacuumSystem) -> tuple[list[Component], dict]:
        solver_components = []
        component_map = {}
        for component in vacuum_system:
            solver_version = self.convert_component(component)
            append_or_extend_to_list(solver_version, solver_components)
            component_map[component] = solver_version
        return solver_components, component_map


def total_conductance(components: list[Component], mass_gas) -> float:
    return 1 / sum([1 / component.C(mass_gas) for component in components])


def make_Q_vec(solver_vac_sys: SolverVacuumSystem) -> np.ndarray:
    Q_vec = np.array([chamber.Q for chamber in solver_vac_sys.chambers_unsolved_pressure()])
    for chamber_a in solver_vac_sys.chambers_unsolved_pressure():
        assert not (chamber_a.P is not None and chamber_a.Q > 0.0)
        idx_a = solver_vac_sys.matrix_index[chamber_a]
        branches = solver_vac_sys.branches(chamber_a)
        for branch in branches:
            assert len(branch) != 1  # either no branch, or at least one tube, then a vacuum chamber
            if len(branch) != 0:
                tubes_and_aps, chamber_b = branch[:-1], branch[-1]
                assert is_all_tubes_and_aps(tubes_and_aps) and type(chamber_b) is Chamber
                if chamber_b.P is not None:
                    C_total = total_conductance(tubes_and_aps, solver_vac_sys.gas_mass)
                    Q_vec[idx_a] = C_total * chamber_b.P
    return Q_vec


def is_all_tubes_and_aps(components: Iterable[Component]) -> bool:
    return all(type(component) in (Tube, Aperture) for component in components)


def make_C_matrix(solver_vac_sys: SolverVacuumSystem) -> np.ndarray:
    C_matrix = np.zeros((len(solver_vac_sys.chambers_unsolved_pressure()),) * 2)
    for chamber_a in solver_vac_sys.chambers_unsolved_pressure():
        idx_a = solver_vac_sys.matrix_index[chamber_a]
        C_matrix[idx_a, idx_a] += chamber_a.S(m_Daltons=solver_vac_sys.gas_mass)
        branches = solver_vac_sys.branches(chamber_a)
        for branch in branches:
            assert len(branch) != 1  # either no branch, or at least one tube, then a vacuum chamber
            if len(branch) != 0:
                tubes_and_aps, chamber_b = branch[:-1], branch[-1]
                assert is_all_tubes_and_aps(tubes_and_aps) and type(chamber_b) is Chamber
                C_total = total_conductance(tubes_and_aps, solver_vac_sys.gas_mass)
                C_matrix[idx_a, idx_a] += C_total
                if chamber_b.P is None:
                    idx_b = solver_vac_sys.matrix_index[chamber_b]
                    C_matrix[idx_a, idx_b] += -C_total
    return C_matrix


def map_pressure_to_tube(tube: Tube, tube_split_components: list[Component]) -> None:
    pressure_profile = [comp.P for comp in tube_split_components if type(comp) is Chamber]
    assert len(tube.P) == len(pressure_profile)
    tube.P[:] = pressure_profile


def update_vacuum_system_with_results(vacuum_system: VacuumSystem, solver_vac_sys: SolverVacuumSystem) -> None:
    for component in vacuum_system:
        if type(component) is Chamber:
            component.P = solver_vac_sys.component_map[component].P
        elif type(component) is Tube:
            tube_split_components = solver_vac_sys.component_map[component]
            map_pressure_to_tube(component, tube_split_components)


def mean_pressure_in_system(vac_sys: VacuumSystem) -> float:
    current_x = 0
    P_vals = []
    P_x_vals = []
    for component in vac_sys:
        if type(component) is Chamber:
            P_vals.append(component.P)
            P_x_vals.append(current_x)
        elif type(component) is Tube:
            P_vals.extend(component.P)
            P_x_vals.extend(component.P_x_vals + current_x)
            current_x += component.L
    assert is_ascending(P_x_vals)
    P_integral = np.trapz(P_vals, x=P_x_vals)
    P_mean = P_integral / (max(P_x_vals) - min(P_x_vals))
    return P_mean


def solve_vac_system(vacuum_system: VacuumSystem) -> None:
    solver_vac_sys = SolverVacuumSystem(vacuum_system)
    Q_mat = make_Q_vec(solver_vac_sys)
    C = make_C_matrix(solver_vac_sys)
    P = np.linalg.inv(C) @ Q_mat
    for chamber in solver_vac_sys.chambers_unsolved_pressure():
        idx = solver_vac_sys.matrix_index[chamber]
        chamber.P = P[idx]

    update_vacuum_system_with_results(vacuum_system, solver_vac_sys)
    vacuum_system.P_mean = mean_pressure_in_system(vacuum_system)


def show_vac_sys(vac_sys: VacuumSystem, fig_size=None, save_fig=None, dpi=300) -> None:
    current_x = 0
    P_vals = []
    P_x_vals = []
    fig, axs = plt.subplots(2, sharex=True, figsize=fig_size)

    fig.suptitle('Vacuum system simulation')
    offset_up = True
    for component in vac_sys:
        if type(component) is Chamber:
            axs[1].scatter(current_x, 0, c='red', marker='s', s=100, label='pump')
            if component.name != 'unassigned':
                y, rotation = (.005, 30) if offset_up else (-.02, -30)
                axs[1].annotate(component.name, [current_x, y], rotation=rotation)
                offset_up = not offset_up
            P_vals.append(component.P)
            P_x_vals.append(current_x)
        elif type(component) is Tube:
            axs[1].plot([current_x, current_x + component.L], [0.0, 0.0], c='black', label='tube')
            P_vals.extend(component.P)
            P_x_vals.extend(component.P_x_vals + current_x)
            current_x += component.L
    assert is_ascending(P_x_vals)
    assert not np.any(np.isnan(P_vals))
    axs[0].semilogy(P_x_vals, P_vals)
    axs[0].grid()
    axs[1].set_xlabel("Position, cm")
    axs[0].set_ylabel("Pressure, Torr")
    axs[1].set_yticks([])
    handles, labels = plt.gca().get_legend_handles_labels()  # avoid redundant labels
    my_label = dict(zip(labels, handles))
    plt.legend(my_label.values(), my_label.keys())
    plt.subplots_adjust(hspace=.0)
    plt.tight_layout()
    if save_fig is not None:
        plt.savefig(save_fig, dpi=dpi)
    plt.show()


def vacuum_lifetime(P: float, gas='H2') -> float:
    """Vacuum lifetime from collisions"""
    P = 133 * P
    n = P / (BOLTZMANN_CONSTANT * ROOM_TEMPERATURE)  # convert from Torr to Pascal
    n = 1e-6 * n  # convert from cubic meter to cubic cm
    K = rate_coefficients[gas]  # rate constant
    tau = 1 / (K * n)
    return tau
