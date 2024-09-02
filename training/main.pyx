import os
import joblib
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


from training.trainer import train
from training.post_processor import save_json, inverse_transform
from training.data_processor import (
	scale_data,
	get_datasets,
	preprocess_data,
	create_sequences
)

from training.model_builder import (
	gru_model,
	lstm_model,
	lstm_gru_model
)


from warnings import filterwarnings
filterwarnings('ignore')

async def main(algorithm: str, sequence_length: int, epochs: int, batch_size: int):
    datasets_path = './datasets'
    models_path   = './models'
    posttrained   = './posttrained'
    pickle_file   = './pickles'


    for dataset in await get_datasets(datasets_path):
        print(f"[TRAINING] {dataset.replace('.csv', '')} ")

        dataframe = pd.read_csv(os.path.join(datasets_path, dataset), index_col='Date')[['Close']]
        model_file = os.path.join(models_path, f"{dataset.replace('.csv', '')}.keras")

        # dataframe = preprocess_data(dataframe)
        dataframe.dropna(inplace = True)
        standard_scaler, dataframe = await scale_data(dataframe, StandardScaler)
        minmax_scaler, dataframe = await scale_data(dataframe, MinMaxScaler)

        sequences, labels = await create_sequences(dataframe, sequence_length)
        input_shape = (sequences.shape[1], sequences.shape[2])

        if algorithm == "GRU":
            model = await gru_model(input_shape)

        elif algorithm == "LSTM":
            model = await lstm_model(input_shape)

        elif algorithm == "LSTM_GRU":
            model = await lstm_gru_model(input_shape)

        else: model = await lstm_model(input_shape)

        train_size = int(len(sequences) * 0.8)
        X_train, X_test = sequences[:train_size], sequences[train_size:]
        y_train, y_test = labels[:train_size], labels[train_size:]

        await train({
			'model': model,
			'model_file': model_file,
			'sequence_length': sequence_length,
			'epochs': epochs,
			'batch_size': batch_size
		}, X_train, y_train, X_test, y_test)

        dataframe_json = {'Date': dataframe.index.tolist(), 'Close': dataframe['Close'].tolist()}

        await save_json(
          os.path.join(posttrained, f'{dataset.replace(".csv", "")}-posttrained.json'),
          dataframe_json
        )

        joblib.dump(minmax_scaler, os.path.join(pickle_file, f'{dataset.replace(".csv", "")}_minmax_scaler.pickle'))
        joblib.dump(standard_scaler, os.path.join(pickle_file, f'{dataset.replace(".csv", "")}_standard_scaler.pickle'))

        model.load_weights(model_file)
        model.save(model_file)

        print("\n\n")

