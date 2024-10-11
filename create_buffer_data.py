from train import CollectBufferData, training_kwargs, CSTREnv
cbd = CollectBufferData(**training_kwargs)
cbd.extend_buffer_data(extend_amount=1000)