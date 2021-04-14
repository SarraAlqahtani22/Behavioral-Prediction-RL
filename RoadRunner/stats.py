import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model



colored = np.load("history_RR_State_Conv2D.npy", allow_pickle=True).item()
bw = np.load("history_RR_State_Conv2D150.npy", allow_pickle=True).item()


plt.plot(colored['val_mae'], label='60')
plt.plot(bw['val_mae'], label='150')
plt.title("RR State Validation MAE")
plt.legend()
plt.show()

print(colored['val_mae'][-5:])
print(bw['val_mae'][-5:])
