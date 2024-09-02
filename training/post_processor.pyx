import json
from sklearn.preprocessing import MinMaxScaler

from warnings import filterwarnings
filterwarnings('ignore')


""" Inverse Transform """
async def inverse_transform(object scaler, data):
	return scaler.inverse_transform(data)


""" save json """
async def save_json(str filename, data):
	with open(filename, 'w') as f:
		json.dump(data, f)

