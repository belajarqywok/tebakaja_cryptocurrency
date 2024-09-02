import os
import numpy as np

from warnings import filterwarnings
filterwarnings('ignore')


""" Get Datasets """
async def get_datasets(str datasets_path):
  cdef list items = os.listdir(datasets_path)
  cdef list csv_files = []
  cdef str item

  for item in items:
    if os.path.isfile(os.path.join(datasets_path, item)) and item.endswith('.csv'):
      csv_files.append(item)
    
  return sorted(csv_files)


""" Create Sequences """
async def create_sequences(df, int sequence_length):
  cdef list labels = []
  cdef list sequences = []
  cdef int i

  for i in range(len(df) - sequence_length):
    seq   = df.iloc[i:i + sequence_length].values
    label = df.iloc[i + sequence_length].values[0]
    sequences.append(seq)
    labels.append(label)
    
  return np.array(sequences), np.array(labels)


""" Pre-Process Data """
async def preprocess_data(dataframe):
  cdef str col

  for col in dataframe.columns:
    if dataframe[col].isnull().any():
      if dataframe[col].dtype == 'object':
        dataframe[col].fillna(dataframe[col].mode()[0], inplace=True)
      else:
        dataframe[col].fillna(dataframe[col].mean(), inplace=True)

  return dataframe


""" Scale Data """
async def scale_data(dataframe, scaler_cls):
  scaler = scaler_cls()
  dataframe['Close'] = scaler.fit_transform(dataframe[['Close']])
  return scaler, dataframe

