import argparse
import os

from azureml.core import Run
from helper.model import compile_fit, get_data_generators, get_model, load_data



def main():
    # Get parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_folder',
        type=str,
        dest='input_folder',
        default="prepped_data",
        help='input folder'
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        dest='output_folder',
        default="weather_model",
        help='output folder'
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of Model",
        default="weather-predict.h5",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        help="Basic Name of Dataset",
        default="weather",
    )
    parser.add_argument(
        "--recurrent_model_type",
        type=str,
        help="Type of Neural Network Model",
        default="LSTM",
    )
    parser.add_argument(
        "--recurrent_units",
        type=int,
        help="Number of recurrent units",
        default=8,
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        help="Sequence Length. Should be Larger than Number of Recurrent Units.",
        default=14,
    )
    parser.add_argument(
        "--optimizer_name",
        type=str,
        help="Name of the optimizer",
        default="Nadam",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning Rate",
        default=0.001,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch Size",
        default=128,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of Maximum Epochs to Train",
        default=50,
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        help="Number of Epochs to Monitor for Early Stopping",
        default=5,
    )
    args = parser.parse_args()

    print("Argument [input_folder]: %s" % args.input_folder)
    print("Argument [output_folder]: %s" % args.output_folder)
    print("Argument [data_name]: %s" % args.data_name)
    print("Argument [model_name]: %s" % args.model_name)
    print("Argument [recurrent_model_type]: %s" % args.recurrent_model_type)
    print("Argument [recurrent_units]: %s" % args.recurrent_units)
    print("Argument [sequence_length]: %s" % args.sequence_length)
    print("Argument [optimizer_name]: %s" % args.optimizer_name)
    print("Argument [learning_rate]: %s" % args.learning_rate)
    print("Argument [batch_size]: %s" % args.batch_size)
    print("Argument [epochs]: %s" % args.epochs)
    print("Argument [early_stopping_patience]: %s" % args.early_stopping_patience)

    output_folder = args.output_folder
    input_folder = args.input_folder
    data_name = args.data_name
    model_name = args.model_name
    if not model_name.endswith('.h5'):
        model_name += '.h5'
    recurrent_model_type = args.recurrent_model_type
    recurrent_units = args.recurrent_units
    sequence_length = args.sequence_length
    optimizer_name = args.optimizer_name
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    patience = args.early_stopping_patience

    # Get the experiment run context
    run = Run.get_context()

    data = load_data(path=input_folder, name=data_name)
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
        run=run
    )

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        model.save(os.path.join(output_folder, model_name))
        print("Model saved successfully.")

    run.complete()


if __name__ == '__main__':
    main()
