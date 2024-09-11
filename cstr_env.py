import numpy as np
from scipy.integrate import odeint

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


class CSTREnv:
    def __init__(self, seed: int | None = None, **kwargs) -> None:
        if seed is None:
            self.seed = np.random
        else:
            self.seed = np.random.RandomState(seed)
        self.__dict__.update(**kwargs)
        self.reset()

    def reset(self):
        self.state = {
            "current_Ca": self.init_Ca // 0.01 / 100,
            "current_T": self.init_T // 0.01 / 100,
            "current_Tc": self.init_Tc // 0.01 / 100,
            "ideal_Ca": self.ideal_Ca,
            "ideal_T": self.ideal_T,
        }
        self.Ca_traj = [self.init_Ca]
        self.T_traj = [self.init_T]
        self.Tc_traj = [self.init_Tc]

    def step(self, action: float, return_xy: bool = False):
        # going on to the new state and calculate reward
        y = odeint(
            func=cstr,
            y0=(self.Ca_traj[-1], self.T_traj[-1]),
            t=[0, 1],  # 1 time frame later
            args=(self.Tc_traj[-1],),
        )

        new_Ca = (
            y[-1][0] + self.noise * self.seed.uniform(low=-1, high=1, size=1) * 0.1
        ).item()
        new_T = (
            y[-1][1] + self.noise * self.seed.uniform(low=-1, high=1, size=1) * 5
        ).item()
        new_Tc = np.clip(
            a=self.Tc_traj[-1] + action, a_max=self.upper_Tc, a_min=self.lower_Tc
        )

        reward = -100 * (
            abs((self.ideal_Ca - new_Ca) / self.ideal_Ca)
            + abs((self.ideal_T - new_T) / self.ideal_T)
        )

        # update state

        self.state = {
            "current_Ca": new_Ca // 0.01 / 100,
            "current_T": new_T // 0.01 / 100,
            "current_Tc": new_Tc // 0.01 / 100,
            "ideal_Ca": self.ideal_Ca,
            "ideal_T": self.ideal_T,
        }

        self.Ca_traj.append(new_Ca)
        self.T_traj.append(new_T)
        self.Tc_traj.append(new_Tc)

        if return_xy:
            return (reward, new_Ca, new_T, new_Tc)
        else:
            return reward
