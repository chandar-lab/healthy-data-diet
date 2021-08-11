# Main script for gathering args.

import argparse
from experiment import run_experiment
from models.baselines.baseline_1 import train_baseline_1
from models.baselines.baseline_2 import train_baseline_2
from models.classifier import measure_bias_metrics
from argparse import ArgumentParser


def parse_args():
    """Parses the command line arguments."""
    parser = ArgumentParser()
    # choosing between our work and the baselines
    parser.add_argument(
        "--method",
        choices=["ours","baseline_1", "baseline_2","baseline_3"],
        default="ours",
        help="Choosing between our work and some of the baseline methods",
    )    
    parser.add_argument(
        "--approach",
        choices=["policy_gradient","supervised_learning"],
        default="policy_gradient",
        help="Choosing between supervised learning and policy gradient approaches",
    )        
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
        "--lambda_gender",
        type=float,
        default=4,
        help="The hyperparameter controling the weight given to the reward due to gender bias",
    )
    parser.add_argument(
        "--lambda_data",
        type=float,
        default=4,
        help="The hyperparameter controling the weight given to the reward due to unintended correlation bias",
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
    if(args.method == "ours"):
      # Measure the performance of our model
      model, tokenizer = run_experiment(args)
      measure_bias_metrics(model, tokenizer, args)

    elif(args.method == "baseline_1"):
      # Measure the perfrmance of the first baseline, which is increases the size of the dataset by gender flipping (data augmentation)
      model, tokenizer = train_baseline_1(args)
      measure_bias_metrics(model, tokenizer, args) 

    elif(args.method == "baseline_2"):
      # Measure the perfrmance of the second baseline, which is explained here https://arxiv.org/pdf/1911.03861.pdf
      model, tokenizer = train_baseline_2(args)
      measure_bias_metrics(model, tokenizer, args)                    
