from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from warnings import filterwarnings
filterwarnings('ignore')


""" Trainer """
async def train(dict configuration, X_train, y_train, X_test, y_test):
	cdef object early_stopping = EarlyStopping(
		monitor = 'val_loss',
		patience = 5,
		mode = 'min'
	)

	cdef object model_checkpoint = ModelCheckpoint(
		filepath = configuration['model_file'],
		save_best_only = True,
		monitor = 'val_loss',
		mode = 'min'
	)

	cdef object history = configuration['model'].fit(
		X_train, y_train,
		epochs = configuration['epochs'],
		batch_size = configuration['batch_size'],
		validation_data = (X_test, y_test),
		callbacks = [ early_stopping, model_checkpoint ]
	)

	return history
    
