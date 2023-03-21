"""Test suite"""

from vacuum_analyzer import VacuumSystem,solve_system,N2_mass,ROOM_TEMPERATURE
from math import isclose,sqrt

def test_1():
    """Simple case of a single chamber with a pump and a gas load. N2 at room temp"""
    Q=25.0
    S=10.0
    P_test=Q/S
    vac_sys = VacuumSystem()
    vac_sys.add_chamber(S=S, Q=Q)
    chamber=vac_sys.chambers()[0]
    P=solve_system(vac_sys)[chamber]
    assert isclose(P,P_test)

def test_2():
    """Simple case of a gas load connected to a pump through a vacuum tube. Check the pressure at the gas load source
    and at the vacuum pump. N2 at room temp. Recall that a short tube model is used"""
    Q=1.0
    S=5.0
    L=10
    D=2.0

    C_tube=(D ** 3 / L) * (3.81 * sqrt(ROOM_TEMPERATURE / N2_mass))
    C_ap=2.9 * sqrt(ROOM_TEMPERATURE / N2_mass) * D ** 2
    vac_sys=VacuumSystem()
    vac_sys.add_chamber(Q=Q)
    vac_sys.add_tube(L,D)
    vac_sys.add_chamber(S)
    P_vals=list(solve_system(vac_sys).values())
    P_gas_load,P_pump=P_vals

    assert isclose(P_pump,Q/S) #pressure above pump

    S_total=1/(1/C_tube +1/C_ap+ 1/S) #total pump speed including tube for gas load. Short tube model
    assert isclose(P_gas_load,Q/S_total,abs_tol=1e-2,rel_tol=1e-2) #pressure at gas leak.


