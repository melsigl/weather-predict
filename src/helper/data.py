import pandas as pd
import numpy as np


def split_ts(dataset, train_ratio=0.6, val_ratio=0.2):
    # training, validation, and testing data creation must not be performed by randomly selecting
    # samples as it would ignore any sequential and time dependent information.
    num_samples = dataset.shape[0]
    train = dataset.iloc[:int(train_ratio * num_samples)]
    val = dataset.iloc[int(train_ratio * num_samples):int((train_ratio + val_ratio) * num_samples)]
    if train_ratio + val_ratio < 1:
        test = dataset.iloc[int((train_ratio + val_ratio) * num_samples):]
        return train, val, test
    return train, val


def clean_data(df):
    # remove leading whitespaces in column names
    df.rename(
        columns={name: name.strip() for name in df.columns.values},
        inplace=True
    )

    # drop duplicates as we have an overlap
    # history file usually spans until end of 2020,
    # whereas recent file contains data from mid of 2020 until yesterday
    df.drop_duplicates(subset='MESS_DATUM', keep='first', inplace=True)

    # drop columns. these columns represent quality levels, we will not need them
    # however, we could only keep rows that satisfy a specific quality level
    df.drop(labels=['QN_4', 'QN_3', 'STATIONS_ID'], axis=1, inplace=True)

    # missing measurements are indicated by -999,
    # thus we replace them with NaN (not a number)
    df.replace(-999, np.nan, inplace=True)

    # and then we just delete all rows containing NaNs
    df.dropna(axis=0, inplace=True)

    # convert MESS_DATUM from int to string and then to date
    df['MESS_DATUM'] = pd.to_datetime(df['MESS_DATUM'].astype(str))

    # set index to our date column
    df.index = pd.DatetimeIndex(df['MESS_DATUM'])

    # generate a date range from min to max to allocate missing dates
    date_range = pd.date_range(
        start=df.MESS_DATUM.min(),
        end=df.MESS_DATUM.max()
    )
    missing_dates = date_range.difference(df.index)
    print(f"Number of missing dates: {len(missing_dates)}")

    # fill in missing dates, missing values are set to NaN
    df = df.reindex(date_range)

    # fill in NaNs with linear interpolation
    # we can also use a different interpolation method,
    # but we will keep it simple here
    df.interpolate(method='linear', inplace=True, axis=0)

    # TMK holds the average daily temperature. we want to predict this,
    # so we add it as an additional column called TARGET
    df['TARGET'] = df['TMK']

    # we are interested in predicting the agerate temperature for tomorrow,
    # thus we shift the temperature target column by one
    df['TARGET'] = df['TARGET'].shift(-1)

    # last target value is nan, thus we exclude the last row:
    df = df.iloc[:-1, :]

    # add months as binary columns
    binaries = pd.DataFrame(0, index=df.index, columns=[f'MONTH{i}' for i in range(1, 13)])
    df = pd.concat([binaries, df], axis=1)

    for i in range(1, binaries.shape[1] + 1):
        df.loc[df.index.month == i, f'MONTH{i}'] = 1

        # we do not need MESS_DATUM anymore.
    # Our model does not care about it, it got no meaning.
    # For our model this column is just growing
    df.drop(labels=['MESS_DATUM'], axis=1, inplace=True)

    return df