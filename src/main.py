import argparse
import distutils
import os
import sys

from finetuning_model import Finetuner
from preprocessing import Preprocessor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

if __name__ == '__main__':

    # Request arguments from user
    parser = argparse.ArgumentParser(description='Simple text classification pipeline')
    parser.add_argument(
        '--train_path',
        type=str,
        nargs='?',
        help='Path to your training data.',
        default=r'../data/train_set_multilabel1.csv',
    )
    parser.add_argument(
        '--valid_path',
        type=str,
        nargs='?',
        help='Path to your valid data.',
        default=r'../data/val_set_multilabel1.csv',
    )
    parser.add_argument(
        '--label_names',
        nargs='+',
        help='The name of your label column(s), can be one or many.',
        default=['label1', 'label2'],
    )
    parser.add_argument(
        '--text_name',
        type=str,
        nargs='?',
        help='The name of your text column.',
        default='text',
    )
    parser.add_argument(
        '--model_path',
        type=str,
        nargs='?',
        help='path to your HuggingFace pretrained model',
        default=r'microsoft/Multilingual-MiniLM-L12-H384',
    )
    parser.add_argument(
        '--is_testing',
        type=lambda x: bool(distutils.util.strtobool(x)),
        nargs='?',
        default=False,
        help='Test your model with 10 samples before running full batch.',
    )
    parser.add_argument(
        '--return_api',
        type=lambda x: bool(distutils.util.strtobool(x)),
        nargs='?',
        default=False,
        help='True if u you want to return API for checking the results.',
    )
    args = parser.parse_args()

    # Change config based on parser arguments
    config.TRAIN_PATH = args.train_path
    config.VAL_PATH = args.valid_path
    config.LABEL_FEATURE_NAMES = args.label_names
    config.TEXT_FEATURE_NAME = args.text_name
    config.MODEL_PATH = args.model_path
    is_testing = args.is_testing
    return_api = args.return_api

    # Preprocessing
    preprocessor = Preprocessor()
    preprocessor.preprocess()

    preprocessed_train = preprocessor.train
    preprocessed_valid = preprocessor.valid

    # Finetuning
    finetuning = Finetuner(
        preprocessed_train, preprocessed_valid, is_testing=is_testing
    )
    finetuning.train_model()
    finetuning.evaluate()

    # Return API
    if return_api:

        # Get the latest checkpoint for API
        checkpoint_path = (
            'model_checkpoints/'
            + sorted(next(os.walk('./model_checkpoints'))[1][:-1])[-1]
        )
        import create_api

        create_api.run_api(checkpoint_path)
