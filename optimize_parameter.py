import configparser
import json

import matplotlib.pyplot as plt

from pid import PIDController
from simulate_data import online_ctrl_Tc

config = configparser.ConfigParser()
config.read("config.ini")


def optimize_parameter(
    current_Kps: list = [
        float(v) for v in json.loads(config["pid"]["Kps"])["init_value"]
    ],
    current_Kis: list = [
        float(v) for v in json.loads(config["pid"]["Kis"])["init_value"]
    ],
    current_Kds: list = [
        float(v) for v in json.loads(config["pid"]["Kds"])["init_value"]
    ],
    current_Kb=float(json.loads(config["pid"]["Kb"])["init_value"]),
    current_Ca: float = float(config["experiment"]["init_Ca"]),
    current_T: float = float(config["experiment"]["init_T"]),
    current_Tc: float = float(config["experiment"]["init_Tc"]),
    ideal_Ca: float = float(config["experiment"]["ideal_Ca"]),
    ideal_T: float = float(config["experiment"]["ideal_T"]),
    upper_Tc: float = float(config["experiment"]["upper_Tc"]),
    lower_Tc: float = float(config["experiment"]["lower_Tc"]),
    weight_e: float = float(config["optimize"]["weight_e"]),
    weight_u: float = float(config["optimize"]["weight_u"]),
    weight_ud: float = float(config["optimize"]["weight_ud"]),
    experiment_step: int = int(config["experiment"]["experiment_step"]),
):
    pid_controller = PIDController(
        Kps=current_Kps,
        Kis=current_Kis,
        Kds=current_Kds,
        Kb=current_Kb,
    )
    loss_f = 0
    pre_delta_Tc = 0
    for _ in range(experiment_step):

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

        # loss f calculation
        loss_f += (
            weight_e * abs(current_Ca - ideal_Ca + current_T - ideal_T)
            + weight_u * abs(delta_Tc)
            + weight_ud * abs(delta_Tc - pre_delta_Tc)
        )

        pre_delta_Tc = delta_Tc

        return loss_f
