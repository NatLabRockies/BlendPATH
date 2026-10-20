from cantera import Solution, Water, gas_constant
from importlib_resources import files

gas_file = files("BlendPATH.network.pipeline_components").joinpath("gasphase.yaml")

gas = Solution(gas_file)
R_GAS = gas_constant

water = Water()
# Set liquid water state, with vapor fraction x = 0
water.TQ = 298.15, 0
h_liquid = water.h
# Set gaseous water state, with vapor fraction x = 1
water.TQ = 298.15, 1
h_gas = water.h

h_water = h_liquid - h_gas
