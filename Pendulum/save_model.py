import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import LSTM, Dense, Input
from tensorflow.python.keras.models import Model, load_model, Sequential
import tensorflow as tf


model = load_model('Pend_State_LSTM.keras')
model.save('Pend_State_LSTM.h5')
reconstructed = load_model('Pend_State_LSTM.h5')
print(model.summary())
print(reconstructed.summary())
