import numpy as np
import os

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.layers import LSTM, Dense, GRU, RNN


class LogMetricToAzure(Callback):
    def __init__(self, run):
        super().__init__()
        self.run = run

    def on_epoch_end(self, epoch, logs=None):
        """Log all metric values to Azure ML experiment run"""
        for metric, value in logs.items():
            self.run.log(metric, value)
            self.run.parent.log(metric, value)


def get_metrics():
    return [
        tf.metrics.MeanSquaredError(name='mse'),
        tf.metrics.MeanAbsoluteError(name='mae'),
        tf.metrics.MeanAbsolutePercentageError('mape'),
        tf.metrics.RootMeanSquaredError(name='rmse'),
        tf.metrics.MeanSquaredLogarithmicError(name='msle')
    ]


def get_callbacks(run, patience=10):
    callbacks = []
    if run is not None:
        callbacks.append(LogMetricToAzure(run))
    if patience is not None:
        callbacks.append(EarlyStopping(monitor='val_mae', patience=patience))
    return callbacks


def get_optimizer(name, learning_rate):
    if name.upper() == 'NADAM':
        return tf.optimizers.Nadam(learning_rate=learning_rate)
    else:
        raise NotImplementedError('Specified Optimizer Name not Implemented')


def compile_fit(model, train, val, patience, epochs, optimizer_name, learning_rate, run):
    model.compile(
        optimizer=get_optimizer(name=optimizer_name, learning_rate=learning_rate),
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=get_metrics()
    )
    model.summary()
    history = model.fit(
        x=train,
        epochs=epochs,
        validation_data=val,
        callbacks=get_callbacks(
            patience=patience,
            run=run
        ),
        verbose=2  # 0 = silent, 1 = progress bar, 2 = one line per epoch
    )
    return history


def get_model(train, recurrent_model_type, recurrent_units):
    x, _ = train[0]
    model = Sequential()
    if recurrent_model_type.upper() == 'LSTM':
        model.add(
            LSTM(
                recurrent_units,
                input_shape=x.shape[1:],
                activation="tanh",
                recurrent_activation="sigmoid",
                recurrent_dropout=0.0,
                use_bias=True,
                unroll=False,
                return_sequences=False
            )
        )
    elif recurrent_model_type.upper() == 'GRU':
        # TODO
        pass
    elif recurrent_model_type.upper() == 'RNN':
        # TODO
        pass
    else:
        raise ValueError('Specify an appropriate recurrent model type: LSTM, GRU, or RNN')
    model.add(Dense(4, activation='elu'))
    model.add(Dense(1))
    return model


def load_data(path, name):
    return {
        key: np.load(f'{os.path.join(path, name)}_{key}.npy')
        for key in ['train', 'val', 'test']
    }


def get_data_generators(data, sequence_length, batch_size):
    return {
        key: TimeseriesGenerator(
            data=dataset[:, :-1],
            targets=dataset[:, -1],
            length=sequence_length,
            batch_size=batch_size,
            shuffle=True if key != 'test' else False
        )
        for key, dataset in data.items()
    }
