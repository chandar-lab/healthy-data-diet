# Main script for gathering args.
from models.classifier import measure_bias_metrics
from argparse import ArgumentParser
import torch
from torch.optim import Adam
from models.data_loader import data_loader
from transformers import BertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """Parses the command line arguments."""
    parser = ArgumentParser()
    # choosing between our work and the baselines
    parser.add_argument(
        "--method",
        choices=[
            "ours",
            "baseline_data_augmentation",
            "baseline_forgettable_examples",
            "baseline_mind_the_tradeoff",
        ],
        default="ours",
        help="Choosing between our work and some of the baseline methods",
    )
    parser.add_argument(
        "--approach",
        choices=["policy_gradient", "supervised_learning"],
        default="policy_gradient",
        help="Choosing between supervised learning and policy gradient approaches",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Samples per batch")
    parser.add_argument(
        "--compute_majority_and_minority_accuracy",
        type=bool,
        default=False,
        help="Whether or not to compute the test accuracy on the majority and minority groups of the test data",
    )
    # arguments for the classifier for pretraining
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
        "--num_epochs_pretraining",
        type=int,
        default=1,
        help="Number of training epochs for the classifier",
    )
    parser.add_argument(
        "--batch_size_pretraining",
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
        "--norm",
        choices=["l1", "l2"],
        default="l2",
        help="Type of distance used to compute the bias reward",
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
    train_dataset, val_dataset, test_dataset = data_loader(args)
    model_name = args.classifier_model
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(set(train_dataset.labels)),
        output_attentions=True,
    )
    model = model.to(device)
    model.load_state_dict(
        torch.load(
            "./saved_models/"
            + args.classifier_model
            + "_"
            + args.method
            + "_"
            + args.approach
            + "_"
            + args.dataset
            + "_"
            + args.norm
            + "_debiased_best.pt",
            map_location=device,
        )
    )
    measure_bias_metrics(model, args, run=0)
