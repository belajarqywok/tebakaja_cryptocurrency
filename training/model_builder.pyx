from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout

from warnings import filterwarnings
filterwarnings('ignore')


""" GRU (Gated Recurrent Units) Model """
async def gru_model(input_shape):
  cdef object model = Sequential([
    GRU(50, return_sequences = True, input_shape = input_shape),
    Dropout(0.2),

    GRU(50, return_sequences = True),
    Dropout(0.2),

    GRU(50, return_sequences = True),
    Dropout(0.2),

    GRU(50, return_sequences = False),
    Dropout(0.2),

    Dense(units = 1)
  ])

  model.compile(optimizer = 'nadam', loss = 'mean_squared_error')
  return model


""" LSTM (Long Short-Term Memory) Model """
async def lstm_model(input_shape):
  cdef object model = Sequential([
    LSTM(50, return_sequences = True, input_shape = input_shape),
    Dropout(0.2),

    LSTM(50, return_sequences = True),
    Dropout(0.2),

    LSTM(50, return_sequences = True),
    Dropout(0.2),

    LSTM(50, return_sequences = False),
    Dropout(0.2),

    Dense(units = 1)
  ])

  model.compile(optimizer = 'nadam', loss = 'mean_squared_error')
  return model


"""
  LSTM (Long Short-Term Memory) and
  GRU (Gated Recurrent Units) Model
"""
async def lstm_gru_model(input_shape):
  cdef object model = Sequential([
    LSTM(50, return_sequences = True, input_shape = input_shape),
    Dropout(0.2),

    GRU(50, return_sequences = True),
    Dropout(0.2),

    LSTM(50, return_sequences = True),
    Dropout(0.2),

    GRU(50, return_sequences = False),
    Dropout(0.2),

    Dense(units = 1)
  ])
        
  model.compile(optimizer = 'nadam', loss = 'mean_squared_error')
  return model
