#LIBRARIES
import matplotlib.pyplot as plt
import numpy as np

history1 = np.load("history_Ant_State_LSTM.npy", allow_pickle=True).item()
history2 = np.load("history_Ant_State_LSTM_v2.npy", allow_pickle=True).item()
history3 = np.load("history_Ant_State_LSTM_v3.npy", allow_pickle=True).item()

#PLOTS
plt.plot(history3['val_mae'], label='v3')
plt.plot(history2['val_mae'], label='v2')
plt.plot(history1['val_mae'], label='v1')

plt.title("Ant State MAE")
plt.legend()
plt.show()


print(history1['val_mae'][-8:])
print(history2['val_mae'][-8:])
print(history3['val_mae'][-8:])