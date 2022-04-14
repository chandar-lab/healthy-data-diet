from model.data_loader import data_loader
from utils import find_biased_examples, log_topk_attention_tokens
from importance_scores import (
    compute_GradN,
    compute_El2N_fairness,
    compute_forgetting_score,
    compute_El2N_performance,
)
from utils import compute_confidence_and_variability
import pandas as pd
from transformers import BertForSequenceClassification, RobertaForSequenceClassification
from transformers import BertTokenizer, RobertaTokenizer
import torch
import re
import numpy as np
import os

# from gender_bender import gender_bend
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softmax = torch.nn.Softmax(dim=1).to(device)


def analyze_results(
    dataset,
    CDA_examples_ranking,
    data_augmentation_ratio,
    data_diet_examples_ranking,
    data_diet_factual_ratio,
    data_diet_counterfactual_ratio,
    data_substitution_ratio,
    max_length,
    classifier_model,
    compute_importance_scores,
    num_epochs_pretraining,
    batch_size_pretraining,
    output_dir,
    model_dir,
    batch_size,
    analyze_attention,
    use_amulet,
    num_epochs_importance_score,
    num_epochs_confidence_variability,
    num_tokens_logged,
    method,
):
    """
    Analyze the results on the validation data by focusing on:
    1) Attention weights: We log the top k tokens to which the classification token (CLS) attends before and after de-biasing.
    2) Type of examples: We follow the procedure in https://arxiv.org/pdf/2009.10795.pdf where the examples
        are categorized into "easy-to-learn", "hard-to-learn" and "ambiguous". The intuition is to know which category is mostly affected by the de-biasing algorithm.
    args:
        dataset: the dataset used
        CDA_examples_ranking: the ranking of the CDa examples
        data_augmentation_ratio: The ratio of data augmentation that we apply, given that the debiasing is using data augmentation
        data_diet_examples_ranking: Type of rankings we use to pick up the examples in data pruning.
        data_diet_factual_ratio: The ratio of the factual examples that we train on while using data diet.
        data_diet_counterfactual_ratio: The ratio of the counterfactual examples that we train on while using data diet.
        data_substitution_ratio: The ratio of the dataset examples that are flipped in data substitution.
        max_length: The maximum length of the sentences that we classify (in terms of the number of tokens)
        classifier_model: the model name
        compute_importance_scores: whether or not to compute the importance scores
        num_epochs_pretraining: the number of epochs for the pretraining (training the biased model)
        batch_size_pretraining: the batch size for the pretraiing (training the biased model)
        output_dir: the output directory that contains the results
        model_dir: the Directory to the model
        batch_size: the batch size for the training of the debiased model
        analyze_attention: whether or not to copmute the distribution of the attention weights
        use_amulet: whether or not to use Microsoft cluster
        num_epochs_importance_score: the number of epochs that we consider for copmuting the importance scores EL2N and GradN
        num_epochs_confidence_variability: the number fo epochs that we consider while computing the confidence and variability
        num_tokens_logged: the top k tokens  that we consider, which the CLS tokens attends to
        method: the debiasing method used
    returns:
        the function doesnt return anything, since the output is written in a csv file.
    """
    train_dataset, val_dataset, test_dataset = data_loader(
        dataset,
        CDA_examples_ranking,
        data_augmentation_ratio,
        data_diet_examples_ranking,
        data_diet_factual_ratio,
        data_diet_counterfactual_ratio,
        data_substitution_ratio,
        max_length,
        classifier_model,
    )

    if compute_importance_scores:
        data_train = pd.read_csv("./data/" + dataset + "_train_original_gender.csv")
        if CDA_examples_ranking == "El2N_fairness":
            # We can only compute the fairness importance scores model is trained for more than 1 epoch.
            # It is computed on the training dataset, as in https://arxiv.org/pdf/2107.07075.pdf

            El2N_fairness_mean = compute_El2N_fairness(
                batch_size_pretraining,
                classifier_model,
                output_dir,
                model_dir,
                use_amulet,
                num_epochs_importance_score,
                batch_size,
                train_dataset,
            )

            data_train[CDA_examples_ranking] = list(
                El2N_fairness_mean.cpu().detach().numpy()
            )

        elif CDA_examples_ranking == "El2N_performance":
            El2N_performance_mean = compute_El2N_performance(
                batch_size_pretraining,
                classifier_model,
                output_dir,
                model_dir,
                use_amulet,
                num_epochs_importance_score,
                batch_size,
                train_dataset,
            )

            data_train[CDA_examples_ranking] = list(
                El2N_performance_mean.cpu().detach().numpy()
            )

        elif CDA_examples_ranking == "forgetting_scores":
            forgetting_scores = compute_forgetting_score(
                batch_size_pretraining,
                classifier_model,
                output_dir,
                model_dir,
                use_amulet,
                batch_size,
                train_dataset,
                num_epochs_pretraining,
            )

            data_train[CDA_examples_ranking] = list(
                forgetting_scores.cpu().detach().numpy()
            )

        elif CDA_examples_ranking == "GradN":
            GradN = compute_GradN(
                batch_size_pretraining,
                classifier_model,
                output_dir,
                model_dir,
                use_amulet,
                num_epochs_importance_score,
                train_dataset,
            )

            data_train[CDA_examples_ranking] = list(GradN.cpu().detach().numpy())

    for split in ["valid", "test"]:
        data = pd.read_csv("./data/" + dataset + "_" + split + "_original_gender.csv")

        if split == "valid":
            dataset_split = val_dataset
        elif split == "test":
            dataset_split = test_dataset

        if num_epochs_pretraining > 1 and split == "valid":
            # We can only compute the confidence and variability when the biased model is trained for more than 1 epoch.
            # It is computed on the validation dataset
            (
                confidence_before_debiasing,
                variability_before_debiasing,
            ) = compute_confidence_and_variability(
                batch_size_pretraining,
                classifier_model,
                output_dir,
                model_dir,
                use_amulet,
                batch_size,
                num_epochs_pretraining,
                num_epochs_confidence_variability,
                train_dataset,
                val_dataset,
            )

            data["confidence"] = list(
                confidence_before_debiasing.cpu().detach().numpy()
            )
            data["variability"] = list(
                variability_before_debiasing.cpu().detach().numpy()
            )

        checkpoint_steps = int(train_dataset.__len__() / batch_size_pretraining)

        if use_amulet:
            output_dir = f"{os.environ['AMLT_OUTPUT_DIR']}/" + output_dir

            model_dir = f"{os.environ['AMLT_OUTPUT_DIR']}/" + model_dir

        Path(model_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        model_checkpoint_path = model_dir + "/checkpoint-" + str(checkpoint_steps)

        if classifier_model in [
            "bert-base-cased",
            "bert-large-cased",
            "distilbert-base-cased",
            "bert-base-uncased",
            "bert-large-uncased",
            "distilbert-base-uncased",
        ]:
            tokenizer = BertTokenizer.from_pretrained(
                "./saved_models/cached_tokenizers/" + classifier_model
            )
            huggingface_model = BertForSequenceClassification
        elif classifier_model in ["roberta-base", "distilroberta-base"]:
            tokenizer = RobertaTokenizer.from_pretrained(
                "./saved_models/cached_tokenizers/" + classifier_model
            )
            huggingface_model = RobertaForSequenceClassification

        model_before_debiasing = huggingface_model.from_pretrained(
            model_checkpoint_path,
            num_labels=len(set(train_dataset.labels)),
            # We only need the attention weights if we are going to analyze the results
            output_attentions=analyze_attention,
        )

        model_before_debiasing = model_before_debiasing.to(device)

        # Load the model that has the de-biased model
        model_before_debiasing.load_state_dict(
            torch.load(
                model_dir + classifier_model + "_" + dataset + "_biased_best.pt",
                map_location=device,
            )
        )

        number_of_labels = len(set(dataset_split.labels))
        prediction_before_debiasing = torch.ones([0, number_of_labels]).to(device)
        for i in range(int(np.ceil(len(dataset_split) / batch_size))):
            with torch.no_grad():
                prediction = model_before_debiasing.forward(
                    input_ids=torch.tensor(
                        dataset_split.encodings["input_ids"][
                            i * batch_size : (i + 1) * batch_size
                        ]
                    ).to(device),
                    attention_mask=torch.tensor(
                        dataset_split.encodings["attention_mask"][
                            i * batch_size : (i + 1) * batch_size
                        ]
                    ).to(device),
                )["logits"]

            predictions_batch = softmax(
                torch.cat(
                    [
                        torch.unsqueeze(prediction[j], dim=0)
                        for j in range(len(prediction))
                    ]
                )
            ).to(device)
            prediction_before_debiasing = torch.cat(
                (prediction_before_debiasing, predictions_batch), 0
            ).to(device)

        y_pred_before_debiasing = torch.argmax(
            prediction_before_debiasing,
            axis=1,
        )
        # =======================================================
        # Load the model after debiasing
        model_after_debiasing = huggingface_model.from_pretrained(
            classifier_model,
            num_labels=len(set(train_dataset.labels)),
            output_attentions=analyze_attention,
        )
        model_after_debiasing = model_after_debiasing.to(device)

        model_after_debiasing.load_state_dict(
            torch.load(
                model_dir
                + classifier_model
                + "_"
                + method
                + "_"
                + dataset
                + "_debiased_best.pt",
                map_location=device,
            )
        )

        prediction_after_debiasing = torch.ones([0, number_of_labels]).to(device)
        for i in range(int(np.ceil(len(dataset_split) / batch_size))):
            with torch.no_grad():
                prediction = model_after_debiasing.forward(
                    input_ids=torch.tensor(
                        dataset_split.encodings["input_ids"][
                            i * batch_size : (i + 1) * batch_size
                        ]
                    ).to(device),
                    attention_mask=torch.tensor(
                        dataset_split.encodings["attention_mask"][
                            i * batch_size : (i + 1) * batch_size
                        ]
                    ).to(device),
                )["logits"]

            predictions_batch = softmax(
                torch.cat(
                    [
                        torch.unsqueeze(prediction[j], dim=0)
                        for j in range(len(prediction))
                    ]
                )
            ).to(device)
            prediction_after_debiasing = torch.cat(
                (prediction_after_debiasing, predictions_batch), 0
            ).to(device)
        # Get the output of the model after debiasing
        y_pred_after_debiasing = torch.argmax(prediction_after_debiasing, axis=1)

        if analyze_attention:
            data = log_topk_attention_tokens(
                batch_size,
                num_tokens_logged,
                data,
                model_before_debiasing,
                model_after_debiasing,
                dataset_split,
                tokenizer,
            )

        # We compute the predictions of a simple logistic regression model, where
        # we consider the biased examples to be the ones that the mode predicts with
        # very high/low p(y|x)
        prediction_logistic_reg = find_biased_examples(dataset, data)
        # ===================================================#

        # To analyze our results, we keep track of the confidence and variability in prediction of each example in the validation data, as well as whether or not
        # it is correctly classified before and after de-biasing.
        ground_truth_labels = torch.tensor(dataset_split.labels).to(device)
        data["Correct classification? before debiasing"] = (
            ground_truth_labels.cpu() == y_pred_before_debiasing.cpu()
        )
        data["Correct classification? after debiasing"] = (
            ground_truth_labels.cpu() == y_pred_after_debiasing.cpu()
        )

        for k in range(prediction_before_debiasing.shape[1]):
            data["p(y=1|x) our model before debiasing for label " + str(k)] = list(
                prediction_before_debiasing[:, k].cpu().detach().numpy()
            )
            data["p(y=1|x) our model after debiasing for label " + str(k)] = list(
                prediction_after_debiasing[:, k].cpu().detach().numpy()
            )
            data["p(y=1|x) logistic regression model for label " + str(k)] = list(
                prediction_logistic_reg[:, k]
            )
        data["number of tokens"] = data[data.columns[0]].apply(
            lambda x: len(re.findall(r"\w+", x))
        )

        if use_amulet:
            output_dir = f"{os.environ['AMLT_OUTPUT_DIR']}/" + output_dir

        file_directory = output_dir + "analysis/"

        Path(file_directory).mkdir(parents=True, exist_ok=True)
        data.to_csv(
            file_directory
            + split
            + "_data_analysis_"
            + dataset
            + "_"
            + method
            + "_"
            + classifier_model
            + ".csv",
            index=False,
        )

    if compute_importance_scores:
        # If the ranking of the examples is not random, we save the importance scores

        data_train.to_csv(
            file_directory
            + "train_data_analysis_"
            + dataset
            + "_"
            + method
            + "_"
            + classifier_model
            + ".csv",
            index=False,
        )
