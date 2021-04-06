#LIBRARIES
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

history_dense = np.load("history_Pend_Action_Dense.npy", allow_pickle=True).item()
history_lstm = np.load("history_Pend_Action_LSTM.npy", allow_pickle=True).item()

#PLOTS
plt.plot(history_dense['val_mae'], label='Val Dense')
plt.plot(history_dense['mae'], label='Train Dense')
plt.plot(history_lstm['val_mae'], label='Val LSTM')
plt.plot(history_lstm['mae'], label='Train LSTM')
plt.title("MAE")
plt.legend()
plt.show()
