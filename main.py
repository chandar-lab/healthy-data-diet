# Main script for gathering args.
from model.metrics import assess_performance_and_bias
from argparse import ArgumentParser
from analysis import analyze_results
import zipfile
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from pathlib import Path
import torch
import os
from model.classifier import train_biased_classifier
from transformers import BertForSequenceClassification, RobertaForSequenceClassification
from model.data_loader import data_loader
import wandb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softmax = torch.nn.Softmax(dim=1).to(device)


def parse_args():
    """Parses the command line arguments."""
    parser = ArgumentParser()
    # choosing between our work and the baselines
    parser.add_argument(
        "--method",
        choices=[
            "data_augmentation",
            "data_substitution",
            "data_balancing",
            "blindness",
            "data_diet",
        ],
        default="data_substitution",
        help="Choosing between our work and some of the baseline methods. CLP stands for Counterfactual Logit Pairing",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Samples per batch")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=16,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="The seed that we are running. We normally run every experiment for 5 seeds.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="learning rate for the Bert classifier",
    )
    parser.add_argument(
        "--classifier_model",
        choices=[
            "bert-base-cased",
            "bert-base-uncased",
            "bert-large-cased",
            "bert-large-uncased",
            "roberta-base",
            "distilroberta-base",
            "distilbert-base-cased",
            "distilbert-base-uncased",
        ],
        default="bert-base-uncased",
        help="Type of classifier used",
    )
    parser.add_argument(
        "--dataset",
        choices=[
            "Jigsaw",
            "Wiki",
            "Twitter",
            "HateBase",
            "EEEC",
        ],
        default="Twitter",
        help="Type of dataset used",
    )
    parser.add_argument(
        "--num_epochs_pretraining",
        type=int,
        default=1,
        help="Number of pretraining epochs for the classifier, which precedes the debiasing (i.e. the number of epochs for the biased model training).",
    )
    parser.add_argument(
        "--num_epochs_confidence_variability",
        type=int,
        default=5,
        help="Number of training epochs that we consider for computing the confidence and variability scores",
    )
    parser.add_argument(
        "--num_epochs_importance_score",
        type=float,
        default=1,
        help="Number of training epochs that we consider for computing the El2N and GradN importance scores. Following the paper, we set of to 10% of the number of epochs needed for convergence",
    )
    parser.add_argument(
        "--batch_size_pretraining",
        type=int,
        default=32,
        help="Batch size for the classifier during pretraining (i.e. during training the biased model).",
    )
    parser.add_argument(
        "--load_biased_classifier",
        type=bool,
        default=False,
        help="Whether or not to load a pretrained classifier",
    )
    parser.add_argument(
        "--compute_importance_scores",
        type=bool,
        default=False,
        help="Whether or not to compute EL2N importance scores from https://arxiv.org/pdf/2107.07075.pdf",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="The maximum length of the sentences that we classify (in terms of the number of tokens)",
    )
    parser.add_argument(
        "--model_checkpoint_path",
        default="./saved_models/checkpoint-",
        help="Path to the saved Bert classifier checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        default="output/",
        help="Directory to the output",
    )
    parser.add_argument(
        "--model_dir",
        default="saved_models/",
        help="Directory to saved models",
    )
    parser.add_argument(
        "--use_wandb",
        type=bool,
        default=True,
        help="Whether or not to use wandb to visualize the results",
    )
    parser.add_argument(
        "--use_amulet",
        type=bool,
        default=False,
        help="Whether or not to run the code on Amulet, which is the cluster used at Microsoft research",
    )
    parser.add_argument(
        "--CDA_examples_ranking",
        choices=[
            "El2N_fairness",
            "El2N_performance",
            "forgetting_scores",
            "GradN",
            "random",
        ],
        default="random",
        help="Type of rankings we use to pick up the examples in CDA. We choose form the EL2N score for performance/GradN in https://arxiv.org/pdf/2107.07075.pdf, our EL2N score for fairness which we propose, random ranking, or forgetting scores https://arxiv.org/pdf/1812.05159.pdf",
    )
    parser.add_argument(
        "--data_diet_examples_ranking",
        choices=[
            "healthy_El2N",
            "healthy_forgetting_scores",
            "healthy_GradN",
            "fairness_only_diet",
            "El2N",
            "forgetting_scores",
            "GradN",
            "random",
            "healthy_random",
            "super_healthy_random",
            "unhealthy_random",
        ],
        default="random",
        help="Type of rankings we use to pick up the examples in data pruning. We choose form the EL2N score in https://arxiv.org/pdf/2107.07075.pdf, our score (which cares about both performance and fairness, thus called healthy diet), or random ranking",
    )
    parser.add_argument(
        "--data_substitution_position",
        choices=[
            "beginning",
            "end",
            "everywhere",
        ],
        default="everywhere",
        help="The position of the examples that are flipped in data substitution. It can be only for the tokens in the begnning/end of the sentence, or just everywhere (the default)",
    )
    parser.add_argument(
        "--data_substitution_ratio",
        type=float,
        default=0,
        help="The ratio of the dataset examples that are flipped in data substitution. It is set to 0.5 in the original CDS paper",
    )
    parser.add_argument(
        "--data_augmentation_ratio",
        type=float,
        default=1,
        help="The ratio of the dataset examples that are flipped in data augmentation. It is set to 1 in the original CDA paper",
    )
    parser.add_argument(
        "--data_diet_factual_ratio",
        type=float,
        default=1,
        help="The ratio of the factual examples that we train on while using data diet.",
    )
    parser.add_argument(
        "--data_diet_counterfactual_ratio",
        type=float,
        default=1,
        help="The ratio of the counterfactual examples that we train on while using data diet.",
    )
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
        "--num_tokens_logged",
        type=int,
        default=5,
        help="The value of k given that we log the top k tokens that the classification token attends to",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert (args.data_diet_counterfactual_ratio != 0 or args.data_diet_factual_ratio != 0)
    # We cannot have both the data_diet_factual_ratio and data_diet_counterfactual_ratio be zero, because this means there is no training data.
    with zipfile.ZipFile("./bias_datasets.zip", "r") as zip_ref:
        zip_ref.extractall("./data")

    if args.use_wandb:
        wandb.init(
            name=str(args.dataset),
            project="Analyzing data-based gender bias mitigation",
            config=args,
        )

    model_dir = args.model_dir
    output_dir = args.output_dir
    
    if args.use_amulet:
        model_dir = f"{os.environ['AMLT_OUTPUT_DIR']}/" + model_dir
        output_dir = f"{os.environ['AMLT_OUTPUT_DIR']}/" + output_dir

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if args.load_biased_classifier == False:
        # Train/Load the biased model using cross-entropy
        # To-do: I should either remove the if or put something in the else
        biased_model = train_biased_classifier(
            args.dataset,
            args.CDA_examples_ranking,
            args.data_augmentation_ratio,
            args.data_diet_examples_ranking,
            args.data_diet_factual_ratio,
            args.data_diet_counterfactual_ratio,
            args.data_substitution_ratio,
            args.max_length,
            args.classifier_model,
            args.batch_size_pretraining,
            model_dir,
            args.use_amulet,
            args.num_epochs_pretraining,
            args.learning_rate,
            args.seed,
        )

        # save the best biased model
        torch.save(
            biased_model.state_dict(),
            model_dir + args.classifier_model + "_" + args.dataset + "_biased_best.pt",
        )

    if args.classifier_model in [
        "bert-base-cased",
        "bert-large-cased",
        "distilbert-base-cased",
        "bert-base-uncased",
        "bert-large-uncased",
        "distilbert-base-uncased",
    ]:
        huggingface_model = BertForSequenceClassification
    elif args.classifier_model in ["roberta-base", "distilroberta-base"]:
        huggingface_model = RobertaForSequenceClassification

    # Load the dataset
    train_dataset, val_dataset, test_dataset = data_loader(
        args.seed,
        args.dataset,
        args.CDA_examples_ranking,
        args.data_augmentation_ratio,
        args.data_diet_examples_ranking,
        args.data_diet_factual_ratio,
        args.data_diet_counterfactual_ratio,
        args.data_substitution_ratio,
        args.max_length,
        args.classifier_model,
        apply_data_augmentation=(args.method == "data_augmentation"),
        apply_data_substitution=(args.method == "data_substitution"),
        apply_blindness=(args.method == "blindness"),
        apply_data_diet=(args.method == "data_diet"),
        apply_data_balancing=(args.method == "data_balancing"),
    )

    # Define pretrained tokenizer and model
    model_name = args.classifier_model
    model = huggingface_model.from_pretrained(
        model_name,
        num_labels=len(set(train_dataset.labels)),
    )

    model = model.to(device)

    # The number of epochs afterwhich we save the model.
    if args.compute_importance_scores:
        # The number of epochs that we consider to compute our fairness score could be less than one to get capture the state of the model in the very early stages of trianing. We save the checkpoint to use them while computing the scores.
        checkpoint_steps = int(train_dataset.__len__() / args.batch_size_pretraining * args.num_epochs_importance_score)
    else:
        # If we are not computing the scores, we can just save the checkpoint after each epoch
        checkpoint_steps = int(train_dataset.__len__() / args.batch_size_pretraining)

    # We now train the model after applying data augmentation/substitution/blindness
    # to the dataset and train from scratch.

    # Define Trainer parameters
    classifier_args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="steps",
        eval_steps=checkpoint_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        load_best_model_at_end=True,
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=classifier_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train the model
    trainer.train()

    torch.save(
        model.state_dict(),
        model_dir
        + args.classifier_model
        + "_"
        + args.method
        + "_"
        + args.dataset
        + "_debiased_best.pt",
    )

    assess_performance_and_bias(
        args.seed,
        model,
        args.dataset,
        args.CDA_examples_ranking,
        args.data_augmentation_ratio,
        args.data_diet_examples_ranking,
        args.data_diet_factual_ratio,
        args.data_diet_counterfactual_ratio,
        args.data_substitution_ratio,
        args.max_length,
        args.classifier_model,
        output_dir,
        model_dir,
        args.use_amulet,
        args.method,
        args.batch_size_pretraining,
        args.batch_size,
        args.use_wandb,
    )
    if args.analyze_results:
        analyze_results(
            args.seed,
            args.dataset,
            args.CDA_examples_ranking,
            args.data_augmentation_ratio,
            args.data_diet_examples_ranking,
            args.data_diet_factual_ratio,
            args.data_diet_counterfactual_ratio,
            args.data_substitution_ratio,
            args.max_length,
            args.classifier_model,
            args.compute_importance_scores,
            args.num_epochs_pretraining,
            args.batch_size_pretraining,
            output_dir,
            model_dir,
            args.batch_size,
            args.analyze_attention,
            args.use_amulet,
            args.num_epochs_importance_score,
            args.num_epochs_confidence_variability,
            args.num_tokens_logged,
            args.method,
        )
