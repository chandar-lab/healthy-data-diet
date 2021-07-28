# Main script for gathering args.

import argparse
from experiment import run_experiment
from classifier import measure_bias_metrics
from argparse import ArgumentParser


def parse_args():
    """Parses the command line arguments."""
    parser = ArgumentParser()
    # arguments for the policy gradient algorithm
    parser.add_argument("--batch_size", type=int, default=64, help="Samples per batch")
    parser.add_argument(
        "--num_epochs_PG",
        type=int,
        default=15,
        help="Number of training epochs for the policy gradient algorithm",
    )
    parser.add_argument(
        "--learning_rate_PG",
        type=float,
        default=1.41e-6,
        help="learning rate for the Bert classifier",
    )
    parser.add_argument(
        "--num_saved_debiased_models",
        type=int,
        default=3,
        help="The number of debiased models that are saved throughout training",
    )
    parser.add_argument(
        "--norm",
        choices=["l1", "l2"],
        default="l2",
        help="Type of distance used to compute the bias reward",
    )
    parser.add_argument(
        "--lambda_PG",
        type=float,
        default=0.01,
        help="The hyperparameter controling the weight given to the reward due to bias and that due to miscalssification",
    )
    parser.add_argument(
        "--compute_majority_and_minority_accuracy",
        type=bool,
        default=False,
        help="Whether or not to compute the test accuracy on the majority and minority groups of the test data",
    )
    # arguments for the classifier
    parser.add_argument(
        "--classifier_model",
        choices=["bert-base-uncased"],
        default="bert-base-uncased",
        help="Type of classifier used",
    )
    parser.add_argument(
        "--dataset",
        choices=[
            "Equity-Evaluation-Corpus",
            "Twitter_sexism_dataset",
            "HASOC_dataset",
            "IMDB_dataset",
            "kindle_dataset",
            "Wikipedia_toxicity_dataset",
            "Twitter_toxicity_dataset",
        ],
        default="twitter_dataset",
        help="Type of dataset used",
    )
    parser.add_argument(
        "--num_epochs_classifier",
        type=int,
        default=1,
        help="Number of training epochs for the classifier",
    )
    parser.add_argument(
        "--batch_size_classifier",
        type=int,
        default=32,
        help="Batch size for the classifier",
    )
    parser.add_argument(
        "--load_pretrained_classifier",
        type=bool,
        default=False,
        help="Whether or not to load a pretrained classifier",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="The maximum length of the sentences that we classify (in terms of the number of tokens)",
    )
    parser.add_argument(
        "--model_path",
        default="./saved_models/checkpoint-",
        help="Path to the saved Bert classifier",
    )
    parser.add_argument(
        "--output_dir",
        default="saved_models",
        help="Directory to saved models",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, tokenizer = run_experiment(args)
    measure_bias_metrics(model, tokenizer, args)
