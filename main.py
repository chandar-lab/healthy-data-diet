# Main script for gathering args.

import argparse
from experiment import run_experiment
from classifier import measure_bias_metrics

parser = argparse.ArgumentParser()


# arguments for the policy gradient algorithm
parser.add_argument("--batch_size", type=int, default=64, help="Samples per batch")
parser.add_argument(
    "--num_epochs",
    type=int,
    default=15,
    help="Number of training epochs for the policy gradient algorithm",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=1.41e-7,
    help="learning rate for the Bert classifier",
)
parser.add_argument(
    "--PG_lambda",
    type=float,
    default=0.01,
    help="The hyperparameter controling the weight given to the reward due to bias and that due to miscalssification",
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
    choices=["Equity-Evaluation-Corpus,twitter_dataset"],
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
    default=512,
    help="The maximum length of the sentences that we classify (in terms of the number of tokens)",
)


def main(args):
    model, tokenizer = run_experiment(args)
    measure_bias_metrics(model, tokenizer, args)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
