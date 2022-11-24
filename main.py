# Main script for gathering args.
from model.metrics import assess_performance_and_bias
from argparse import ArgumentParser
from analysis import analyze_results
import numpy as np
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
        help="Choosing between our work and some of the baseline methods.",
    )
    parser.add_argument(
        "--batch_size_debiased_model", type=int, default=64, help="Samples per batch"
    )
    parser.add_argument(
        "--num_epochs_debiased_model",
        type=int,
        default=1,
        help="Number of training epochs for the training of the debiased model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="The seed that we are running the experiment for. We run every experiment for 5 seeds.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="learning rate for the optimizer",
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
        "--num_epochs_biased_model",
        type=int,
        default=1,
        help="Number of epochs for the training of the biased classifier, which precedes the debiasing.",
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
        help="Number of training epochs that we consider for computing the EL2N and GraNd importance scores. Following the paper, we set it to 10% of the number of epochs needed for convergence",
    )
    parser.add_argument(
        "--batch_size_biased_model",
        type=int,
        default=64,
        help="Batch size for the training of the biased model.",
    )
    parser.add_argument(
        "--compute_importance_scores",
        type=bool,
        default=False,
        help="Whether or not to compute importance scores",
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
        help="Path to the saved classifier checkpoint",
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
        default=False,
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
            "GE",
            "EL2N",
            "forget_score",
            "GraNd",
            "random",
        ],
        default="random",
        help="Type of rankings we use to pick up the examples in data augmentation. We choose form the EL2N/GraNd in https://arxiv.org/pdf/2107.07075.pdf, our GE score for fairness which we propose, random ranking, or forgetting scores https://arxiv.org/pdf/1812.05159.pdf",
    )
    parser.add_argument(
        "--data_diet_examples_ranking",
        choices=[
            "healthy_EL2N",
            "healthy_forget_score",
            "healthy_GraNd",
            "vanilla_GE",
            "EL2N",
            "forget_score",
            "GraNd",
            "healthy_GE",
            "unhealthy_GE",
            "random",
            "healthy_random",
            "super_healthy_random",
            "unhealthy_random",
            "healthy_data_diet",
            "unhealthy_data_diet",
        ],
        default="random",
        help="Type of rankings we use to pick up the examples in data pruning.",
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
        default=True,
        help="Whether or not to analyze the results by finding the examples that flipped from wrongly predicted to correctly predicted, computing the top tokens that the model attends to, and the test and validation performances",
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
    assert args.data_diet_counterfactual_ratio != 0 or args.data_diet_factual_ratio != 0
    # We cannot have both the data_diet_factual_ratio and data_diet_counterfactual_ratio be zero, because this means there is no training data.
    with zipfile.ZipFile("./bias_dataset_arxiv.zip", "r") as zip_ref:
        zip_ref.extractall("./data")

    if args.use_wandb:
        wandb_mode = "online"
    else:
        wandb_mode = "offline"

    if args.seed != None:
        my_seed = args.seed
    else:
        my_seed = np.random.randint(10000, size=1)[0]

    wandb.init(
        name=str(args.dataset),
        project="Deep learning on healthy data diet",
        config=args,
        mode=wandb_mode,
    )

    model_name = args.classifier_model
    model_dir = args.model_dir + "/" + model_name + "/"
    output_dir = args.output_dir

    if args.use_amulet:
        model_dir = f"{os.environ['AMLT_OUTPUT_DIR']}/" + model_dir
        output_dir = f"{os.environ['AMLT_OUTPUT_DIR']}/" + output_dir

    if args.method == "data_diet":
        method_details = (
            args.method
            + "_"
            + args.data_diet_examples_ranking
            + "_"
            + str(args.data_diet_factual_ratio)
            + "_"
            + str(args.data_diet_counterfactual_ratio)
        )
    else:
        method_details = args.method
    output_dir = (
        "./results/arxiv_2/"
        + method_details
        + "_"
        + args.classifier_model
        + "_"
        + args.dataset
        + "_Seed_"
        + str(my_seed)
        + "_CDS_ratio_"
        + str(args.data_substitution_ratio)
        + "_CDA_ratio_"
        + str(args.data_augmentation_ratio)
        + "_"
        + args.CDA_examples_ranking
        + "/"
        + output_dir
    )

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Train/Load the biased model using cross-entropy
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
        args.batch_size_biased_model,
        model_dir,
        args.use_amulet,
        args.num_epochs_biased_model,
        args.learning_rate,
        my_seed,
    )

    # save the best biased model
    torch.save(
        biased_model.state_dict(),
        model_dir
        + "/"
        + args.classifier_model
        + "_"
        + args.dataset
        + "_"
        + args.method
        + "_"
        + args.data_diet_examples_ranking
        + "_"
        + str(args.data_augmentation_ratio)
        + "_"
        + str(args.data_diet_factual_ratio)
        + "_"
        + str(args.data_diet_counterfactual_ratio)
        + "_biased_best.pt",
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
        my_seed,
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
    model = huggingface_model.from_pretrained(
        "./saved_models/cached_models/" + model_name,
        num_labels=len(set(train_dataset.labels)),
    )

    model = model.to(device)

    # The number of epochs after which we save the model.
    if args.compute_importance_scores:
        # The number of epochs that we consider to compute our fairness score could be less than one to get capture the state of the model in the very early stages of trianing. We save the checkpoint to use them while computing the scores.
        checkpoint_steps = int(
            train_dataset.__len__()
            / args.batch_size_biased_model
            * args.num_epochs_importance_score
        )
    else:
        # If we are not computing the scores, we can just save the checkpoint after each epoch
        checkpoint_steps = int(train_dataset.__len__() / args.batch_size_biased_model)

    # We now train the debiased model

    # Define Trainer parameters
    classifier_args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="steps",
        eval_steps=checkpoint_steps,
        save_steps=checkpoint_steps,
        per_device_train_batch_size=args.batch_size_debiased_model,
        per_device_eval_batch_size=args.batch_size_debiased_model,
        num_train_epochs=args.num_epochs_debiased_model,
        learning_rate=args.learning_rate,
        seed=my_seed,
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
        + args.dataset
        + "_"
        + args.method
        + "_"
        + args.data_diet_examples_ranking
        + "_"
        + str(args.data_augmentation_ratio)
        + "_"
        + str(args.data_diet_factual_ratio)
        + "_"
        + str(args.data_diet_counterfactual_ratio)
        + "_debiased_best.pt",
    )

    assess_performance_and_bias(
        my_seed,
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
        args.batch_size_biased_model,
        args.batch_size_debiased_model,
        args.use_wandb,
    )
    if args.analyze_results:
        analyze_results(
            my_seed,
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
            args.num_epochs_biased_model,
            args.batch_size_biased_model,
            output_dir,
            model_dir,
            args.batch_size_debiased_model,
            args.analyze_attention,
            args.use_amulet,
            args.num_epochs_importance_score,
            args.num_epochs_confidence_variability,
            args.num_tokens_logged,
            args.method,
        )
