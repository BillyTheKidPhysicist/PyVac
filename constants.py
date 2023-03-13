"""
Constants such as pump speed and outgassing for use with vacuum modeling.
"""


# outgassing rates, in units of torr L /s, from stainless steel with high temperature heat treating procedure

outgassing_rates = {'H2': 5e-13, 'He': 0.0}
ROOM_TEMPERATURE: float = 293.15  # room temperature in Kelvin
BOLTZMANN_CONSTANT: float = 1.38064852e-23  # Boltzman constant, SI
# different species have different pumping speeds. From Duniway manual for ion pump lineup

# collision rate coefficents
rate_coefficients = {'H2': 6e-9, 'He': 1.46e-9}
ion_speed_factors = {'H2': 2.0, 'He': .1}
_small_ion_pump_speed = 10.0
small_ion_pump_speed = {'H2': ion_speed_factors['H2'] * _small_ion_pump_speed,
                                   'He': ion_speed_factors['He'] * _small_ion_pump_speed}

_big_ion_pump_speed = 200.0
big_ion_pump_speed = {'H2': ion_speed_factors['H2'] * _big_ion_pump_speed,
                                 'He': ion_speed_factors['He'] * _big_ion_pump_speed}
turbo_pump_speed = 100.0
diffusion_pump_speed = 1000.0

big_chamber_pressure = 2e-6
