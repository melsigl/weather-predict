import glob
import zipfile

import pandas as pd

import os
import requests

import unidecode
from azureml.core import Workspace, Dataset


def download(url, path):
    result = requests.get(url, allow_redirects=True)
    open(path, 'wb').write(result.content)


def unzip(zip_file, destination):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(destination)


def minimal_data_preparation(hist_folder, current_folder, csv_path):
    historical_path = glob.glob(os.path.join(hist_folder, 'produkt_klima_tag_*.txt'))[0]
    current_path = glob.glob(os.path.join(current_folder, 'produkt_klima_tag_*.txt'))[0]
    df_hist = pd.read_csv(
        historical_path,
        sep=';'
    )
    df_current = pd.read_csv(
        current_path,
        sep=';'
    )
    df = pd.concat([df_hist, df_current])
    df.drop_duplicates(inplace=True)
    df.drop(labels='eor', inplace=True, axis=1)
    df.to_csv(csv_path, index=False)
    print(csv_path)


def aml_register_dataset(csv_path, csv_file_name):
    ws = Workspace.from_config('config.json')
    datastore = ws.get_default_datastore()
    # upload file to remote datastore
    datastore.upload_files(
        files=[csv_path],
        target_path='weather',
        overwrite=True,
        show_progress=True
    )
    # create a TabularDataset from file path in datastore
    datastore_paths = [(datastore, f'weather/{csv_file_name}.csv')]
    weather_ds = Dataset.Tabular.from_delimited_files(path=datastore_paths)
    weather_ds.register(workspace=ws, name=csv_file_name)


def download_helper(urls, city, delete_zip=True):
    paths = []
    csv_file_name = 'weather-' + unidecode.unidecode(city)
    csv_file_name = csv_file_name.replace('/', '-').lower()
    csv_path = os.path.join('data', csv_file_name + '.csv')
    for url in urls:
        zip_path = os.path.join('data', url.split('/')[-1])
        destination = os.path.splitext(zip_path)[0]
        paths.append(destination)
        download(url, zip_path)
        unzip(zip_path, destination)
        if delete_zip:
            os.remove(zip_path)
    minimal_data_preparation(paths[0], paths[1], csv_path)
    aml_register_dataset(csv_path, csv_file_name)


def main():
    climate_data = {
        'Nürnberg': [
            'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/tageswerte_KL_03668_18790101_20201231_hist.zip',
            'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/recent/tageswerte_KL_03668_akt.zip'
        ],
        'Bamberg': [
            'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/tageswerte_KL_00282_19490101_20201231_hist.zip',
            'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/recent/tageswerte_KL_00282_akt.zip'
        ],
        'Augsburg': [
            'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/tageswerte_KL_00232_19470101_20201231_hist.zip',
            'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/recent/tageswerte_KL_00232_akt.zip'
        ],
        'München': [
            'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/tageswerte_KL_01262_19920517_20201231_hist.zip',
            'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/recent/tageswerte_KL_01262_akt.zip'
        ],
        'Frankfurt/Main': [
            'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/tageswerte_KL_01420_19490101_20201231_hist.zip',
            'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/recent/tageswerte_KL_01420_akt.zip'
        ]
    }
    os.mkdir('data')
    for city, urls in climate_data.items():
        print(f'Downloading files for [{city}]')
        download_helper(urls, city)


if __name__ == "__main__":
    main()
