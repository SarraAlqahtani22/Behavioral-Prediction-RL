#LIBRARIES
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

'''
history_dense = np.load("history_Pend_Action_Dense.npy", allow_pickle=True).item()
history_lstm = np.load("history_Pend_Action_LSTM.npy", allow_pickle=True).item()

#PLOTS
plt.plot(history_dense['val_mae'], label='Val Dense')
plt.plot(history_dense['mae'], label='Train Dense')
plt.plot(history_lstm['val_mae'], label='Val LSTM')
plt.plot(history_lstm['mae'], label='Train LSTM')
plt.title("Action MAE")
plt.legend()
plt.show()

history_dense = np.load("history_Pend_State_Dense.npy", allow_pickle=True).item()
history_lstm = np.load("history_Pend_State_LSTM.npy", allow_pickle=True).item()

#PLOTS
plt.plot(history_dense['val_mae'], label='Val Dense')
plt.plot(history_dense['mae'], label='Train Dense')
plt.plot(history_lstm['val_mae'], label='Val LSTM')
plt.plot(history_lstm['mae'], label='Train LSTM')
plt.title("State MAE")
plt.legend()
plt.show()

'''
history_lstm = np.load("history_Pend_State_LSTM.npy", allow_pickle=True).item()
history_150 = np.load("history_Pend_State_LSTM.npy", allow_pickle=True).item()


plt.plot(history_lstm['val_mae'], label='50')
plt.plot(history_150['val_mae'], label='500 Train')
plt.title("State MAE")
plt.legend()
plt.show()

print(history_lstm['val_mae'][-8:])
print(history_150['val_mae'][-8:])
