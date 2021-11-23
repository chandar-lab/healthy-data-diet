# Main script for gathering args.
from experiment import run_experiment
from model.classifier import assess_performance_and_bias
from argparse import ArgumentParser
from analysis import analyze_results


def parse_args():
    """Parses the command line arguments."""
    parser = ArgumentParser()
    # choosing between our work and the baselines
    parser.add_argument(
        "--method",
        choices=[
            "baseline_data_augmentation",
            "baseline_data_substitution",
            "CLP",
            "baseline_data_substitution",
        ],
        default="ours",
        help="Choosing between our work and some of the baseline methods. CLP stands for Counterfactual Logit Pairing",
    )
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs to do.")
    parser.add_argument(
        "--approach",
        choices=["policy_gradient", "supervised_learning"],
        default="policy_gradient",
        help="Choosing between supervised learning and policy gradient approaches",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Samples per batch")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=16,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1.41e-6,
        help="learning rate for the Bert classifier",
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
            "Jigsaw_toxicity_dataset",
            "Wikipedia_toxicity_dataset",
            "Twitter_sexism_dataset",
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
        "--model_path",
        default="./saved_models/checkpoint-",
        help="Path to the saved Bert classifier",
    )
    parser.add_argument(
        "--output_dir",
        default="saved_models",
        help="Directory to saved models",
    )
    parser.add_argument(
        "--use_wandb",
        type=bool,
        default=True,
        help="Whether or not to use wandb to visualize the results",
    )  
    parser.add_argument(
        "--use_auxiliary_loss",
        type=bool,
        default=False,
        help="Whether or not to use the auxiliary loss that we are proposing",
    )    
    parser.add_argument(
        '--hidden_dropout',
        type=float,
        default=0.1,
        help='Dropout rate for the hidden layers of the model')   
    parser.add_argument(
        '--attention_dropout',
        type=float,
        default=0.1,
        help='Dropout rate for the attention layers of the model')    
    parser.add_argument(
        '--tokenizer_dropout',
        type=float,
        default=0.1,
        help='Dropout rate of the tokenizer')       
    parser.add_argument(
        '--num_hidden_layers',
        type=int,
        default=12,
        help='Number of layers in the model')  
    # arguments for analysing the data
    parser.add_argument(
        "--analyze_results",
        type=bool,
        default=False,
        help="Whether or not to analyze the results by finding the examples that flipped from wrongly predicted to correctly predicted, and computing the top tokens that the model attends to",
    )
    parser.add_argument(
        "--analyze_attention",
        type=bool,
        default=False,
        help="Whether or not to analyze the attention map",
    )    
    parser.add_argument(
        "--log_top_tokens_each_head",
        type=bool,
        default=False,
        help="Whether or not to log the top k tokens that the classification token attends to in the last layer for each attention head",
    )
    parser.add_argument(
        "--num_tokens_logged",
        type=int,
        default=5,
        help="The value of k given that we log the top k tokens that the classification token attends to",
    )
    parser.add_argument(
        "--num_saved_debiased_models",
        type=int,
        default=3,
        help="The number of debiased models that are saved throughout training",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for run in range(args.num_runs):         
        model, tokenizer = run_experiment(args, run)
        assess_performance_and_bias(model, args, run)            
    if args.analyze_results == True:
        analyze_results(args, model)
