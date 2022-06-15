# Main script for gathering args.
from model.metrics import assess_performance_and_bias
from main import parse_args
import torch
from model.data_loader import data_loader
from transformers import BertForSequenceClassification, RobertaForSequenceClassification
from analysis import analyze_results
from pathlib import Path
import wandb
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    args = parse_args()

    if args.use_wandb:
        wandb.init(
            name=str(args.dataset),
            project="Analyzing data-based gender bias mitigation",
            config=args,
        )

    train_dataset, val_dataset, test_dataset = data_loader(
        args.seed,
        dataset=args.dataset,
        CDA_examples_ranking=args.CDA_examples_ranking,
        data_augmentation_ratio=args.data_augmentation_ratio,
        data_diet_examples_ranking=args.data_diet_examples_ranking,
        data_diet_factual_ratio=args.data_diet_factual_ratio,
        data_diet_counterfactual_ratio=args.data_diet_counterfactual_ratio,
        data_substitution_ratio=args.data_substitution_ratio,
        max_length=args.max_length,
        classifier_model=args.classifier_model,
    )
    model_name = args.classifier_model

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

    model = huggingface_model.from_pretrained(
        model_name,
        num_labels=len(set(train_dataset.labels)),
        output_attentions=args.analyze_attention,
    )
    model = model.to(device)

    model_dir = args.model_dir
    output_dir = args.output_dir
    
    if args.use_amulet:
        model_dir = f"{os.environ['AMLT_OUTPUT_DIR']}/" + model_dir
        output_dir = f"{os.environ['AMLT_OUTPUT_DIR']}/" + output_dir

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

 
    model.load_state_dict(
        torch.load(
            model_dir
            + args.classifier_model
            + "_"
            + args.method
            + "_"
            + args.dataset
            + "_debiased_best.pt",
            map_location=device,
        )
    )
    assess_performance_and_bias(
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
