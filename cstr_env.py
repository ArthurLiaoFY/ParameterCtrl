import numpy as np

###############
#  CSTR model #
###############

# Taken from http://apmonitor.com/do/index.php/Main/NonlinearControl


def cstr(x, t, u):

    # ==  Inputs == #
    Tc_value = u  # Temperature of cooling jacket (K)

    # == States == #
    Ca_value, T_value = (
        x  # (Concentration of A in CSTR (mol/m^3),  Temperature in CSTR (K))
    )

    # == Process parameters == #
    Tf = 350  # Feed temperature (K)
    q = 100  # Volumetric Flowrate (m^3/sec)
    Caf = 1  # Feed Concentration (mol/m^3)
    V = 100  # Volume of CSTR (m^3)
    rho = 1000  # Density of A-B Mixture (kg/m^3)
    Cp = 0.239  # Heat capacity of A-B Mixture (J/kg-K)
    mdelH = 5e4  # Heat of reaction for A->B (J/mol)
    EoverR = 8750  # E -Activation energy (J/mol), R -Constant = 8.31451 J/mol-K
    k0 = 7.2e10  # Pre-exponential factor (1/sec)
    UA = 5e4  # U -Heat Transfer Coefficient (W/m^2-K) A -Area - (m^2)

    # == Equations == #
    rA = k0 * np.exp(-EoverR / T_value) * Ca_value  # reaction rate
    dCadt = q / V * (Caf - Ca_value) - rA  # Calculate concentration derivative
    dTdt = (
        q / V * (Tf - T_value)
        + mdelH / (rho * Cp) * rA
        + UA / V / rho / Cp * (Tc_value - T_value)
    )  # Calculate temperature derivative
    # == Equations == #
    rA = k0 * np.exp(-EoverR / T_value) * Ca_value  # reaction rate
    dCadt = q / V * (Caf - Ca_value) - rA  # Calculate concentration derivative
    dTdt = (
        q / V * (Tf - T_value)
        + mdelH / (rho * Cp) * rA
        + UA / V / rho / Cp * (Tc_value - T_value)
    )  # Calculate temperature derivative

    # == Return == #
    return dCadt, dTdt
