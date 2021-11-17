import argparse
import glob
import os

from azureml.core import Run, Model


def main():
    # Get parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_folder',
        type=str,
        dest='input_folder',
        default='input_folder',
        help='input folder'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        help='Name of the Trained and Saved Model',
        default='weather-predict',
    )
    parser.add_argument(
        '--register_model_name',
        type=str,
        help='Name of the Registered Model',
        default='weather-predict',
    )
    parser.add_argument(
        '--model_description',
        type=str,
        dest='model_description',
        default='Model predicting tomorrow\'s average air temperature.',
        help='Registered Model Description.'
    )
    args = parser.parse_args()

    print('Argument [input_folder]: %s' % args.input_folder)
    print('Argument [model_name]: %s' % args.model_name)
    print('Argument [register_model_name]: %s' % args.register_model_name)
    print('Argument [model_description]: %s' % args.model_description)

    input_folder = args.input_folder
    model_name = args.model_name
    register_model_name = args.register_model_name
    model_description = args.model_description

    # Get the experiment run context
    run = Run.get_context()
    workspace = run.experiment.workspace

    model_path = glob.glob(f'{os.path.join(input_folder, model_name)}*.h5')[0]
    print(f'Found model: {model_path}')
    model = Model.register(
        workspace=workspace,
        model_name=register_model_name,
        model_path=model_path,
        description=model_description
    )
    print(f'Registered version {model.version} of model {model.name}')
    run.complete()


if __name__ == '__main__':
    main()
