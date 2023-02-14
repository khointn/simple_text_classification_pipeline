from chitra.serve import create_api
from transformers import pipeline


def run_api(checkpoint_path):
    print('\nRUNNING API...\n')
    classifier = pipeline(
        'text-classification', model=checkpoint_path, return_all_scores=True
    )
    create_api(classifier, run=True, api_type='text-classification')
