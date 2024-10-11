# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import math

import numpy as np
from scipy.integrate import odeint


def cstr_system(y, t, u):
    """
    Differential equations for a continuous stirred-tank reactor model

    t : Time [seconds]
    y : Controlled variables
        C_A : Concentration of reactant A [mol/l]
        C_B : Concentration of reactant B [mol/l]
        T_R : Temperature inside the reactor [Celsius]
        T_K : Temperature of cooling jacker [Celsius]
    u : Manipulated variables
        F : Flow [l/h]
        Q_dot : Heat flow [kW]
    """
    # Process parameters
    K0_ab = 1.287e12  # K0 [h^-1]
    K0_bc = 1.287e12  # K0 [h^-1]
    K0_ad = 9.043e9  # K0 [l/mol.h]
    R_gas = 8.3144621e-3  # Universal gas constant
    E_A_ab = 9758.3 * 1.00  # * R_gas# [kJ/mol]
    E_A_bc = 9758.3 * 1.00  # * R_gas# [kJ/mol]
    E_A_ad = 8560.0 * 1.0  # * R_gas# [kJ/mol]
    H_R_ab = 4.2  # [kJ/mol A]
    H_R_bc = -11.0  # [kJ/mol B] Exothermic
    H_R_ad = -41.85  # [kj/mol A] Exothermic
    Rou = 0.9342  # Density [kg/l]
    Cp = 3.01  # Specific Heat capacity [kJ/Kg.K]
    Cp_k = 2.0  # Coolant heat capacity [kJ/kg.k]
    A_R = 0.215  # Area of reactor wall [m^2]
    V_R = 10.01  # 0.01 # Volume of reactor [l]
    m_k = 5.0  # Coolant mass [kg]
    T_in = 130.0  # Temp of inflow [Celsius]
    K_w = 4032.0  # [kJ/h.m^2.K]
    C_A0 = (
        (5.7 + 4.5) / 2.0 * 1.0
    )  # Concentration of A in input Upper bound 5.7 lower bound 4.5 [mol/l]
    # Process variables
    F, Q_dot = u
    C_a, C_b, T_R, T_K = y
    T_dif = T_R - T_K
    # Rate constants
    K_1 = K0_ab * math.exp((-E_A_ab) / ((T_R + 273.15)))
    K_2 = K0_bc * math.exp((-E_A_bc) / ((T_R + 273.15)))
    K_3 = K0_ad * math.exp((-E_A_ad) / ((T_R + 273.15)))
    # Differential equations
    dC_adt = F * (C_A0 - C_a) - K_1 * C_a - K_3 * (C_a**2)
    dC_bdt = -F * C_b + K_1 * C_a - K_2 * C_b
    dT_Rdt = (
        (
            (K_1 * C_a * H_R_ab + K_2 * C_b * H_R_bc + K_3 * (C_a**2) * H_R_ad)
            / (-Rou * Cp)
        )
        + F * (T_in - T_R)
        + (((K_w * A_R) * (-T_dif)) / (Rou * Cp * V_R))
    )
    dT_Kdt = (Q_dot + K_w * A_R * (T_dif)) / (m_k * Cp_k)

    return dC_adt, dC_bdt, dT_Rdt, dT_Kdt


# def cstr_env(
#     Ca,
#     Cb,
#     Tr,
#     Tk,
#     F,
#     Q_dot,
#     t_start=0,
#     t_end=1,
#     num=5,
# ):
#     odeint(
#         func=cstr_system,
#         y0=(Ca, Cb, Tr, Tk),
#         t=np.linspace(start=t_start, stop=t_end, num=num),
#         args=((F, Q_dot)),
#     )


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
            # -------------------------
            "current_Ca": self.init_Ca,
            "current_Cb": self.init_Cb,
            "current_Tr": self.init_Tr,
            "current_Tk": self.init_Tk,
            "current_F": self.init_F,
            "current_Q": self.init_Q,
            # -------------------------
            # "ideal_Ca": self.ideal_Ca,
            # "ideal_Cb": self.ideal_Cb,
            # "ideal_Tr": self.ideal_Tr,
            # "ideal_Tk": self.ideal_Tk,
        }
        self.Ca_traj = [self.init_Ca]
        self.Cb_traj = [self.init_Cb]
        self.Tr_traj = [self.init_Tr]
        self.Tk_traj = [self.init_Tk]
        self.F_traj = [self.init_F]
        self.Q_traj = [self.init_Q]

    def step(self, action: tuple[float, float], return_xy: bool = False):
        # new action
        new_F = np.clip(
            a=self.F_traj[-1] + action[0],
            a_max=self.upper_F,
            a_min=self.lower_F,
        )
        new_Q = np.clip(
            a=self.Q_traj[-1] + action[1],
            a_max=self.upper_Q,
            a_min=self.lower_Q,
        )

        y = odeint(
            func=cstr_system,
            y0=(
                self.Ca_traj[-1],
                self.Cb_traj[-1],
                self.Tr_traj[-1],
                self.Tk_traj[-1],
            ),
            t=[0, 1],  # 1 time frame later
            args=([new_F, new_Q],),
        )

        # new state
        new_Ca = (
            y[-1][0] + self.noise * self.seed.uniform(low=-1, high=1, size=1) * 0.1
        ).item()
        new_Cb = (
            y[-1][1] + self.noise * self.seed.uniform(low=-1, high=1, size=1) * 0.1
        ).item()
        new_Tr = (
            y[-1][2] + self.noise * self.seed.uniform(low=-1, high=1, size=1) * 5.0
        ).item()
        new_Tk = (
            y[-1][3] + self.noise * self.seed.uniform(low=-1, high=1, size=1) * 5.0
        ).item()

        # reward
        reward = -1 * (
            ((self.ideal_Ca - new_Ca) / self.ideal_Ca) ** 2
            + ((self.ideal_Cb - new_Cb) / self.ideal_Cb) ** 2
            + ((self.ideal_Tr - new_Tr) / self.ideal_Tr) ** 2
            + ((self.ideal_Tk - new_Tk) / self.ideal_Tk) ** 2
        )

        # update state
        self.state = {
            # -------------------------
            "current_Ca": new_Ca,
            "current_Cb": new_Cb,
            "current_Tr": new_Tr,
            "current_Tk": new_Tk,
            "current_F": new_F,
            "current_Q": new_Q,
            # -------------------------
            # "ideal_Ca": self.ideal_Ca,
            # "ideal_Cb": self.ideal_Cb,
            # "ideal_Tr": self.ideal_Tr,
            # "ideal_Tk": self.ideal_Tk,
        }

        self.Ca_traj.append(new_Ca)
        self.Cb_traj.append(new_Cb)
        self.Tr_traj.append(new_Tr)
        self.Tk_traj.append(new_Tk)

        self.F_traj.append(new_F)
        self.Q_traj.append(new_Q)

        if return_xy:
            return (reward, new_Ca, new_Cb, new_Tr, new_Tk, new_F, new_Q)
        else:
            return reward
