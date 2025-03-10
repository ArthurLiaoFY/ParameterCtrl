import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_reward_trend(
    rewards: list | np.ndarray,
    file_path: str = "./plots",
    prefix: str = "",
    suffix: str = "",
    fig_name: str = "reward_trend",
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=np.arange(len(rewards)), y=rewards, mode="lines+markers")
    )
    fig.update_layout(
        title_text="Reward Trend",
        template="plotly_white",
    )

    plotly.offline.plot(
        figure_or_data=fig, filename=f"{file_path}/{prefix}{fig_name}{suffix}.html"
    )


def plot_inference_result(
    inference_traj: dict,
    file_path: str = "./plots",
):
    fig1 = make_subplots(rows=2, cols=2, subplot_titles=("Ca", "Cb", "Tr", "Tk"))
    for target_idx, target_name in zip(range(4), ["Ca", "Cb", "Tr", "Tk"]):

        traj_max_len = -1
        for episode, traj in inference_traj[target_name].items():
            traj_max_len = max(traj_max_len, len(traj))
            fig1.add_trace(
                go.Scatter(
                    y=traj,
                    mode="lines+markers",
                    name=f"{target_name} Traj [{episode}]",
                ),
                row=target_idx // 2 + 1,
                col=target_idx % 2 + 1,
            )
        fig1.add_trace(
            go.Scatter(
                y=[
                    inference_traj.get(f"ideal_{target_name}")
                    for _ in range(traj_max_len)
                ],
                mode="lines",
                name=f"Ideal {target_name}",
                line=dict(dash="dash"),
            ),
            row=target_idx // 2 + 1,
            col=target_idx % 2 + 1,
        )

    fig1.update_layout(
        title_text="Observed Value Trend",
        template="plotly_white",
    )

    fig2 = make_subplots(rows=2, cols=1, subplot_titles=("F", "Q"))
    for target_idx, target_name in zip(range(2), ["F", "Q"]):

        traj_max_len = -1
        for episode, traj in inference_traj[target_name].items():
            traj_max_len = max(traj_max_len, len(traj))
            fig2.add_trace(
                go.Scatter(
                    y=traj,
                    mode="lines+markers",
                    name=f"{target_name} Traj [{episode}]",
                ),
                row=target_idx + 1,
                col=1,
            )

    fig2.update_layout(
        title_text="Input Value Trend",
        template="plotly_white",
    )

    plotly.offline.plot(
        figure_or_data=fig1,
        filename=f"{file_path}/ddpg_inference_observed_value_trend.html",
    )
    plotly.offline.plot(
        figure_or_data=fig2,
        filename=f"{file_path}/ddpg_inference_input_value_trend.html",
    )
