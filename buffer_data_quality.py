# %%
from train import CollectBufferData, training_kwargs

cbd = CollectBufferData(**training_kwargs)
cbd.extend_buffer_data(extend_amount=700)

# %%
a = cbd.replay_buffer.sample(10000)
# %%
a.keys()
# %%
import matplotlib.pyplot as plt

for i in range(6):
    plt.plot(a.get("state").detach().numpy()[:, i], "o")
    plt.show()

# %%
plt.plot(a.get("reward"), "o")
# %%
for i in range(2):
    plt.plot(a.get("action").detach().numpy()[:, i], "o")
    plt.show()
# %%
