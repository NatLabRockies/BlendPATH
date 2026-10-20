import cantera as ct

R = ct.gas_constant / 1000


def calc_a_b(T_c, P_c):
    # From https://thermo.readthedocs.io/thermo.eos.html#redlich-kwong-equations-of-state
    a = 0.42748 * R**2 * T_c ** (2.5) / P_c
    b = 0.08664 * R * T_c / P_c
    return a, b


if __name__ == "__main__":
    t_k = 273.15

    print("CH4", calc_a_b(-82.595 + t_k, 4598800))  # CH4
    print("H2", calc_a_b(-239.95 + t_k, 1297000))  # H2
    print("C2H6", calc_a_b(32.68 + t_k, 4880000))  # C2H6
    print("C3H8", calc_a_b(96.67 + t_k, 4250000))  # C3H8
    print("CO2", calc_a_b(31.05 + t_k, 7386000))  # CO2
    print("H2O", calc_a_b(647, 220.64 * 1e5))  # H2O
    print("O2", calc_a_b(154.58, 50.43 * 1e5))  # O2
    print("N2", calc_a_b(-146.95 + t_k, 3390000))  # N2
    print("n-C4H10", calc_a_b(151.99 + t_k, 3784000))  # n-C4H10
    print("i-C4H10", calc_a_b(134.98 + t_k, 3648000))  # i-C4H10
    print("n-C5H12", calc_a_b(196.54 + t_k, 3364000))  # n-C5H12
