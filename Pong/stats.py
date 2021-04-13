import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model



colored = np.load("history_Pong_State_Conv2D.npy", allow_pickle=True).item()
#bw = np.load("history_Pong_State_Conv2DBW.npy", allow_pickle=True).item()


plt.plot(colored['val_mae'], label='Colored')
#plt.plot(bw['val_mae'], label='Black and White')
plt.title("Pong State Validation MAE")
plt.legend()
plt.show()

print(colored['val_mae'][-5:])
