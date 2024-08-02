import os
import json
import joblib
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



from warnings import filterwarnings
filterwarnings('ignore')

class DataProcessor:
    def __init__(self, datasets_path):
        self.datasets_path = datasets_path
        self.datasets = self._get_datasets()

    def _get_datasets(self):
        return sorted([
            item for item in os.listdir(self.datasets_path)
            if os.path.isfile(os.path.join(self.datasets_path, item)) and item.endswith('.csv')
        ])

    @staticmethod
    def create_sequences(df, sequence_length):
        labels, sequences = [], []
        for i in range(len(df) - sequence_length):
            seq = df.iloc[i:i + sequence_length].values
            label = df.iloc[i + sequence_length].values[0]
            sequences.append(seq)
            labels.append(label)
        return np.array(sequences), np.array(labels)

    @staticmethod
    def preprocess_data(dataframe):
        for col in dataframe.columns:
            if dataframe[col].isnull().any():
                if dataframe[col].dtype == 'object':
                    dataframe[col].fillna(dataframe[col].mode()[0], inplace = True)
                else:
                    dataframe[col].fillna(dataframe[col].mean(), inplace = True)
        return dataframe

    @staticmethod
    def scale_data(dataframe, scaler_cls):
        scaler = scaler_cls()
        dataframe['Close'] = scaler.fit_transform(dataframe[['Close']])
        return scaler, dataframe


class ModelBuilder:
    """
        GRU (Gated Recurrent Units) Model
    """
    @staticmethod
    def gru_model(input_shape):
        model = Sequential([
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


    """
        LSTM (Long Short-Term Memory) Model
    """
    @staticmethod
    def lstm_model(input_shape):
        model = Sequential([
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
    @staticmethod
    def lstm_gru_model(input_shape):
        model = Sequential([
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


class Trainer:
    def __init__(self, model, model_file, sequence_length, epochs, batch_size):
        self.model = model
        self.model_file = model_file
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, X_train, y_train, X_test, y_test):
        early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, mode = 'min')

        model_checkpoint = ModelCheckpoint(
          filepath       = self.model_file,
          save_best_only = True,
          monitor        = 'val_loss',
          mode           = 'min'
        )

        history = self.model.fit(
            X_train, y_train,
            epochs          = self.epochs,
            batch_size      = self.batch_size,
            validation_data = (X_test, y_test),
            callbacks       = [early_stopping, model_checkpoint]
        )

        return history


class PostProcessor:
    @staticmethod
    def inverse_transform(scaler, data):
        return scaler.inverse_transform(data)

    @staticmethod
    def save_json(filename, data):
        with open(filename, 'w') as f:
            json.dump(data, f)

def main(algorithm: str, sequence_length: int, epochs: int, batch_size: int):
    datasets_path = './datasets'
    models_path   = './models'
    posttrained   = './posttrained'
    pickle_file   = './pickles'

    batch_size = 32

    data_processor = DataProcessor(datasets_path)

    for dataset in data_processor.datasets:
        print(f"[TRAINING] {dataset.replace('.csv', '')} ")

        dataframe = pd.read_csv(os.path.join(datasets_path, dataset), index_col='Date')[['Close']]
        model_file = os.path.join(models_path, f"{dataset.replace('.csv', '')}.keras")

        # dataframe = data_processor.preprocess_data(dataframe)
        dataframe.dropna(inplace = True)
        standard_scaler, dataframe = data_processor.scale_data(dataframe, StandardScaler)
        minmax_scaler, dataframe = data_processor.scale_data(dataframe, MinMaxScaler)

        sequences, labels = data_processor.create_sequences(dataframe, sequence_length)
        input_shape = (sequences.shape[1], sequences.shape[2])

        if algorithm == "GRU":
            model = ModelBuilder.gru_model(input_shape)

        elif algorithm == "LSTM":
            model = ModelBuilder.lstm_model(input_shape)

        elif algorithm == "LSTM_GRU":
            model = ModelBuilder.lstm_gru_model(input_shape)

        else: model = ModelBuilder.lstm_model(input_shape)

        train_size = int(len(sequences) * 0.8)
        X_train, X_test = sequences[:train_size], sequences[train_size:]
        y_train, y_test = labels[:train_size], labels[train_size:]

        trainer = Trainer(model, model_file, sequence_length, epochs, batch_size)
        trainer.train(X_train, y_train, X_test, y_test)

        dataframe_json = {'Date': dataframe.index.tolist(), 'Close': dataframe['Close'].tolist()}

        PostProcessor.save_json(
          os.path.join(posttrained, f'{dataset.replace(".csv", "")}-posttrained.json'),
          dataframe_json
        )

        joblib.dump(minmax_scaler, os.path.join(pickle_file, f'{dataset.replace(".csv", "")}_minmax_scaler.pickle'))
        joblib.dump(standard_scaler, os.path.join(pickle_file, f'{dataset.replace(".csv", "")}_standard_scaler.pickle'))

        model.load_weights(model_file)
        model.save(model_file)

        print("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Tebakaja Model Trainer")

    parser.add_argument('-a', '--algorithm', type = str, required = True,
        help = 'select the algorithm to be trained (LSTM, GRU, LSTM_GRU)')

    parser.add_argument('-e', '--epochs', type = int, required = True, help = 'epochs')
    parser.add_argument('-b', '--batchs', type = int, required = True, help = 'batch length')
    parser.add_argument('-s', '--sequences', type = int, required = True, help = 'sequences length')

    args = parser.parse_args()

    main(
        epochs     = args.epochs,
        batch_size = args.batchs,
        algorithm  = args.algorithm,
        sequence_length = args.sequences
    )

