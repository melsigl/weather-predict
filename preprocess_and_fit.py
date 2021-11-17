import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import statsmodels.tsa.api as smt

from src.helper.data import split_ts, clean_data
from src.helper.model import get_model, compile_fit, get_data_generators


def plot(true, predicted, label_true='true', label_predicted='predicted', filename='prediction.pdf'):
    x=true.index
    fig = plt.figure()
    plt.plot(x, true, label=label_true)
    plt.plot(x, predicted, label=label_predicted)
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig(filename)


def plot_history(history):
    logged_metrics = [metric for metric in history.history.keys() if not metric.startswith('val')]

    for logged_metric in logged_metrics:
        fig = plt.figure()
        plt.plot(history.history[logged_metric])
        plt.plot(history.history['val_' + logged_metric])
        plt.title(f'Learning curve for: {logged_metric.upper()}')
        plt.ylabel(logged_metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(f'history_{logged_metric}.pdf')


def residual_analysis(y, lags=None, figsize=(10, 8), filename='residual_analysis.pdf'):
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    hist_ax = plt.subplot2grid(layout, (1, 1))
    sns.lineplot(x=list(range(len(y))), y=y, ax=ts_ax)

    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    sns.distplot(y, ax=hist_ax)
    acf_ax.set_xlim(1.5)
    acf_ax.set_ylim(-0.1, 0.2)
    fig.suptitle(f'Residual Analysis, Residual mean: {np.mean(y):.4f}')
    sns.despine()
    plt.tight_layout()
    plt.savefig(filename)


def main():
    epochs = 100
    learning_rate = 0.001
    batch_size = 128
    sequence_length = 21
    patience = 5
    recurrent_model_type = 'GRU'
    recurrent_units = 8
    optimizer_name = 'NADAM'

    df = pd.read_csv(os.path.join('data', 'weather-nurnberg.csv'))
    df = clean_data(df)

    # _ = df.iloc[:700, :5].plot(subplots=True)
    # plt.show()
    #
    # _ = df.iloc[:700, 5:10].plot(subplots=True)
    # plt.show()
    #
    # _ = df.iloc[:700, 10:-1].plot(subplots=True)
    # plt.show()
    #
    # _ = df.iloc[:700, -1].plot(subplots=True)
    # plt.show()

    train_df, val_df, test_df = split_ts(df, train_ratio=0.8, val_ratio=0.15)
    feature_names = [name for name in df.columns.values if (name != 'TARGET' and not name.startswith('MONTH'))]
    transformer = ColumnTransformer(
        [('features', MinMaxScaler(), feature_names)],
        remainder='passthrough'
    )
    
    data = {
        'train': transformer.fit_transform(train_df),
        'val': transformer.transform(val_df),
        'test': transformer.transform(test_df)
    }

    log_dir = os.path.join(
        "log", str(datetime.today().strftime('%Y-%m-%d'))
    )

    data_generators = get_data_generators(
        data=data,
        sequence_length=sequence_length,
        batch_size=batch_size
    )

    model = get_model(
        train=data_generators['train'],
        recurrent_model_type=recurrent_model_type,
        recurrent_units=recurrent_units,
    )

    _ = compile_fit(
        model=model,
        train=data_generators['train'],
        val=data_generators['val'],
        patience=patience, 
        epochs=epochs,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        run=None
    )

    predictions = model.predict(data_generators['test'])

    plot_dir = 'plot'
    if not os.path.exists('plot'):
        os.makedirs(plot_dir)  
    plot(
        true=test_df.iloc[sequence_length:, -1], 
        predicted=predictions, 
        filename=os.path.join(plot_dir, 'prediction_all.pdf')
    )
    plot(
        true=test_df.iloc[sequence_length:, -1][-180:], 
        predicted=predictions[-180:], 
        filename=os.path.join(plot_dir, 'prediction_last180.pdf')
    )
    plot(
        true=test_df.iloc[sequence_length:sequence_length+180, -1], 
        predicted=predictions[:180], 
        filename=os.path.join(plot_dir, 'prediction_first180.pdf')
    )
    plot(
        true=test_df.iloc[sequence_length:sequence_length+500, -1], 
        predicted=predictions[:500], 
        filename=os.path.join(plot_dir, 'prediction_first500.pdf')
    )
    plot(
        true=test_df.iloc[sequence_length:, -1][-60:], 
        predicted=predictions[-60:], 
        filename=os.path.join(plot_dir, 'prediction_last60.pdf')
    )

    # plot_history(history)

    len(predictions)

    evaluation = model.evaluate(data_generators['test'])
    print(evaluation)

    true = test_df.iloc[sequence_length:, -1].to_numpy()
    pred = np.reshape(predictions, (-1,))
    residuals = true - pred
    residuals.shape
    residual_analysis(
        y=residuals, 
        filename=os.path.join(plot_dir, 'residual_analysis.pdf')
    )

    rmse = mean_squared_error(test_df.iloc[sequence_length:, -1], pred, squared=True)
    mse = mean_squared_error(test_df.iloc[sequence_length:, -1], pred, squared=False)

    print(f'RMSE: {rmse}\nMSE: {mse}')


if __name__ == "__main__":
    main()
