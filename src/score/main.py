import os

from joblib import load
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import load_model
from tensorflow.python.keras.preprocessing.sequence import TimeseriesGenerator

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.standard_py_parameter_type import (
    StandardPythonParameterType,
)

import pandas as pd

from azureml.core.model import Model

from helper.data import clean_data


def init():
    global model
    # we assume that we have just one model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder
    # (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = Model.get_model_path(os.getenv("AZUREML_MODEL_DIR").split("/")[-2])

    model = load_model(model_path)
    print("Model loaded")


# We can specify input and output schemata 
input_sample = StandardPythonParameterType(
    [
        {
            'STATIONS_ID': StandardPythonParameterType(3668),
            'MESS_DATUM': StandardPythonParameterType(20210918),
            'QN_3': StandardPythonParameterType(1),
            'FX': StandardPythonParameterType(8.1),
            'FM': StandardPythonParameterType(2.0),
            'QN_4': StandardPythonParameterType(1),
            'RSK': StandardPythonParameterType(0.0),
            'RSKF': StandardPythonParameterType(0),
            'SDK': StandardPythonParameterType(8.517),
            'SHK_TAG': StandardPythonParameterType(0),
            'NM': StandardPythonParameterType(3.8),
            'VPM': StandardPythonParameterType(12.3),
            'PM': StandardPythonParameterType(978.99),
            'TMK': StandardPythonParameterType(13.6),
            'UPM': StandardPythonParameterType(80.08),
            'TXK': StandardPythonParameterType(20.1),
            'TNK': StandardPythonParameterType(8.0),
            'TGK': StandardPythonParameterType(6.2)
        },
        {
            'STATIONS_ID': StandardPythonParameterType(3668),
            'MESS_DATUM': StandardPythonParameterType(20210918),
            'QN_3': StandardPythonParameterType(1),
            'FX': StandardPythonParameterType(8.1),
            'FM': StandardPythonParameterType(2.0),
            'QN_4': StandardPythonParameterType(1),
            'RSK': StandardPythonParameterType(0.0),
            'RSKF': StandardPythonParameterType(0),
            'SDK': StandardPythonParameterType(8.517),
            'SHK_TAG': StandardPythonParameterType(0),
            'NM': StandardPythonParameterType(3.8),
            'VPM': StandardPythonParameterType(12.3),
            'PM': StandardPythonParameterType(978.99),
            'TMK': StandardPythonParameterType(13.6),
            'UPM': StandardPythonParameterType(80.08),
            'TXK': StandardPythonParameterType(20.1),
            'TNK': StandardPythonParameterType(8.0),
            'TGK': StandardPythonParameterType(6.2)
        }
    ]
)

sample_output_entry = StandardPythonParameterType(
    {
        "MESS_DATUM": StandardPythonParameterType(20210918),
        "AVG_TEMP": StandardPythonParameterType(12.5)
    }
)

output_sample = StandardPythonParameterType(
    {"results": StandardPythonParameterType([sample_output_entry, sample_output_entry])}
)


# Inference_schema generates a schema for your web service
# It then creates an OpenAPI (Swagger) specification for the web service
# at http://<scoring_base_url>/swagger.json
@input_schema("data", input_sample)
@output_schema(output_sample)
def run(data):        
    # load our transformer for feature normalization
    with open('/var/azureml-app/src/score/column_transformer.joblib', 'rb') as f:
        transformer = load(f)
    print("Transformer loaded")

    # TODO set sequence length to the same value you trained your model with
    sequence_length = 21

    print(f"received data: {data}")
    df = pd.DataFrame.from_records(data)
    dates = [date for date in (df.iloc[sequence_length:, :]).loc[:, 'MESS_DATUM']]
    cleaned_data = clean_data(df)

    transformed_data = transformer.transform(cleaned_data)
    ts_gen = TimeseriesGenerator(
            data=transformed_data[:, :-1],
            targets=transformed_data[:, -1],
            length=sequence_length,
            batch_size=128,
            shuffle=False
    )

    predictions = model.predict(ts_gen)
    predictions = [float(prediction[0]) for prediction in predictions]

    print(dates[:5])

    result = [
        {
            "MESS_DATUM": date,
            "avg_temperature": temperature,
        }
        for date, temperature in zip(
            dates, predictions
        )
    ]
    return {"result": result}
