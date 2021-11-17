import argparse
import os
from joblib import dump

import numpy as np
import pandas as pd
from azureml.core import Run, Dataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler

from helper.data import split_ts, clean_data


def main():
    # Get parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_name',
        type=str,
        dest='dataset_name',
        default="weather-nurnberg",
        help='Name of the Weather Dataset'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        dest='train_ratio',
        default=0.8,
        help='Training Dataset Ratio'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        dest='val_ratio',
        default=0.15,
        help='Validation Dataset Ratio'
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        dest='output_folder',
        default="weather_model",
        help='output folder'
    )
    parser.add_argument(
        '--transformer_folder',
        type=str,
        dest='transformer_folder',
        default="transformer",
        help='Datastore location to store min-max scaling values'
    )
    args = parser.parse_args()

    print("Argument [dataset_name]: %s" % args.dataset_name)
    print("Argument [train_ratio]: %s" % args.train_ratio)
    print("Argument [val_ratio]: %s" % args.val_ratio)
    print("Argument [output_folder]: %s" % args.output_folder)
    print("Argument [transformer_folder]: %s" % args.transformer_folder)

    dataset_name = args.dataset_name
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    output_folder = args.output_folder
    transformer_folder = args.transformer_folder

    # Get the experiment run context
    run = Run.get_context()
    ws = run.experiment.workspace

    # load dataset by name
    dataset = Dataset.get_by_name(workspace=ws, name=dataset_name)
    dataset = dataset.to_pandas_dataframe()
    dataset = clean_data(dataset)

    feature_names = [name for name in dataset.columns.values if name != 'TARGET']

    train, val, test = split_ts(dataset, train_ratio=train_ratio, val_ratio=val_ratio)

    transformer = ColumnTransformer(
        [('features', MinMaxScaler(), feature_names)],
        remainder='passthrough'
    )

    train = transformer.fit_transform(train)
    val = transformer.transform(val)
    test = transformer.transform(test)

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        train_output_path = os.path.join(output_folder, f'{dataset_name}_train.npy')
        val_output_path = os.path.join(output_folder, f'{dataset_name}_val.npy')
        test_output_path = os.path.join(output_folder, f'{dataset_name}_test.npy')
        np.save(train_output_path, train)
        np.save(val_output_path, val)
        np.save(test_output_path, test)

    if transformer_folder is not None:
        os.makedirs(transformer_folder, exist_ok=True)
        with open(os.path.join(transformer_folder, 'column_transformer.joblib'), 'wb') as f:
            dump(transformer, f)

    run.complete()


if __name__ == '__main__':
    main()
