import configparser
import json

import matplotlib.pyplot as plt

from pid import PIDController
from simulate_data import online_ctrl_Tc

config = configparser.ConfigParser()
config.read("config.ini")

pid_controller = PIDController(
    Kps=[float(v) for v in json.loads(config["pid"]["Kps"])["init_value"]],
    Kis=[float(v) for v in json.loads(config["pid"]["Kis"])["init_value"]],
    Kds=[float(v) for v in json.loads(config["pid"]["Kds"])["init_value"]],
    Kb=float(json.loads(config["pid"]["Kb"])["init_value"]),
)

current_Ca = float(config["experiment"]["init_Ca"])
current_T = float(config["experiment"]["init_T"])
current_Tc = float(config["experiment"]["init_Tc"])


ideal_Ca = float(config["experiment"]["ideal_Ca"])
ideal_T = float(config["experiment"]["ideal_T"])

upper_Tc = float(config["experiment"]["upper_Tc"])
lower_Tc = float(config["experiment"]["lower_Tc"])

Ca_l = [current_Ca]
T_l = [current_T]
Tc_l = [current_Tc]
loss_f_l = []

for i in range(int(config["experiment"]["experiment_step"])):
    pre_delta_Tc = 0
    if i // float(config["optimize"]["optimize_step"]) == 0:
        if i != 0:
            loss_f_l.append(loss_f)
        loss_f = 0

    current_Ca, current_T, current_Tc, delta_Tc = online_ctrl_Tc(
        controller=pid_controller,
        current_Ca=current_Ca,
        current_T=current_T,
        current_Tc=current_Tc,
        ideal_Ca=ideal_Ca,
        ideal_T=ideal_T,
        upper_Tc=upper_Tc,
        lower_Tc=lower_Tc,
    )
    Ca_l.append(current_Ca)
    T_l.append(current_T)
    Tc_l.append(current_Tc)

    # loss f calculation
    loss_f += (
        float(config["optimize"]["weight_e"])
        * abs(current_Ca - ideal_Ca + current_T - ideal_T)
        + float(config["optimize"]["weight_u"]) * abs(delta_Tc)
        + float(config["optimize"]["weight_ud"]) * abs(delta_Tc - pre_delta_Tc)
    )

    pre_delta_Tc = delta_Tc

# %%
plt.plot(Ca_l, "o")
plt.plot([ideal_Ca] * len(Ca_l), "--", color="r")
plt.show()

plt.plot(T_l, "o")
plt.plot([ideal_T] * len(T_l), "--", color="r")
plt.show()

plt.plot(Tc_l)
plt.plot([upper_Tc] * len(Tc_l), "--", color="r")
plt.plot([lower_Tc] * len(Tc_l), "--", color="r")

plt.show()

plt.plot(loss_f_l, "o-")
plt.plot([0.0] * len(loss_f_l))

plt.show()

# %%
