from train import CollectBufferData, TrainDDPG, training_kwargs

buffer_data = CollectBufferData(**training_kwargs)
# buffer_data.extend_buffer_data(extend_amount=2000)

tddpg = TrainDDPG(**training_kwargs)
tddpg.train_agent(
    buffer_data=buffer_data,
    plot_loss_trend=True,
)
