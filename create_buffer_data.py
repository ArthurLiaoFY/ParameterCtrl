# %%
from train import CollectBufferData, training_kwargs

cbd = CollectBufferData(**training_kwargs)
cbd.extend_buffer_data(extend_amount=700)
