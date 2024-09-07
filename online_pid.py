import configparser
import json

import matplotlib.pyplot as plt

from pid import PIDController
from simulate_data import online_ctrl_Tc

config = configparser.ConfigParser()
config.read("config.ini")

pid_controller = PIDController(
    Kps=[float(v) for v in json.loads(config["pid"]["Kps"])],
    Kis=[float(v) for v in json.loads(config["pid"]["Kis"])],
    Kds=[float(v) for v in json.loads(config["pid"]["Kds"])],
    Kb=float(config["pid"]["Kb"]),
)

current_Ca = float(config["experiment"]["init_Ca"])
current_T = float(config["experiment"]["init_T"])
current_Tc = float(config["experiment"]["init_Tc"])


ideal_Ca = float(config["experiment"]["ideal_Ca"])
ideal_T = float(config["experiment"]["ideal_T"])


Ca_l = [current_Ca]
T_l = [current_T]
Tc_l = [current_Tc]

for _ in range(int(config["experiment"]["time_step"])):
    current_Ca, current_T, current_Tc, _ = online_ctrl_Tc(
        controller=pid_controller,
        current_Ca=current_Ca,
        current_T=current_T,
        current_Tc=current_Tc,
        ideal_Ca=ideal_Ca,
        ideal_T=ideal_T,
    )
    Ca_l.append(current_Ca)
    T_l.append(current_T)
    Tc_l.append(current_Tc)

# %%
plt.plot(Ca_l)
plt.plot([ideal_Ca] * len(Ca_l))
plt.show()

plt.plot(T_l)
plt.plot([ideal_T] * len(T_l))
plt.show()

plt.plot(Tc_l)
plt.show()

# %%
