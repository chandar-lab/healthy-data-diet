from transformers import BertForSequenceClassification, RobertaForSequenceClassification
from model.data_loader import data_loader
from sklearn.metrics import roc_auc_score
from pathlib import Path
import torch
import json
import numpy as np
import torch.nn.functional as F
import pandas as pd
import wandb
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assess_performance_and_bias(
    seed,
    model_after_bias_reduction,
    dataset,
    CDA_examples_ranking,
    data_augmentation_ratio,
    data_diet_examples_ranking,
    data_diet_factual_ratio,
    data_diet_counterfactual_ratio,
    data_substitution_ratio,
    max_length,
    classifier_model,
    output_dir,
    model_dir,
    use_amulet,
    method,
    batch_size_biased_model,
    batch_size_debiased_model,
    use_wandb,
):
    """
    Measure the performance before and after applying our de-biasing algorithm.
    This is done by computing the metrics on the validation, test and IPTTS datasets.
    The IPTTS dataset is a synthetic dataset, developed to measure the bias in the model.
    The metrics that we compute are the following:
    1) AUC
    2) Accuracy
    3) FNED
    4) FPED
    5) TNED
    6) TPED
    7) Demographic parity
    8) Equality of odds
    args:
        seed: the seed_used
        model_after_bias_reduction: the model after updating its weights due to bias reduction
        dataset: the dataset used
        CDA_examples_ranking: the ranking of the CDA examples
        data_augmentation_ratio: The ratio of data augmentation that we apply, given that the debiasing is using data augmentation
        data_diet_examples_ranking: Type of rankings we use to pick up the examples in data pruning.
        data_diet_factual_ratio: The ratio of the factual examples that we train on while using data diet.
        data_diet_counterfactual_ratio: The ratio of the counterfactual examples that we train on while using data diet.
        data_substitution_ratio: The ratio of the dataset examples that are flipped in data substitution.
        max_length: The maximum length of the sentences that we classify (in terms of the number of tokens)
        classifier_model: the model name
        output_dir: the directory to the output
        model_dir: the Directory to the model
        use_amulet: whether or not to run the code on Amulet, which is the cluster used at Microsoft research
        method: the debiasing method used
        batch_size_biased_model: the batch size for the pretraining (training the biased model)
        batch_size_debiased_model: the batch size for the training of the debiased model
        use_wandb: whether or not to use wandb

        the function doesnt return anything, since all the metrics are saved in json files.
    """
    all_metrics = []
    # We need to load the datasets on which we measure the metrics.
    (
        train_dataset,
        val_dataset,
        test_dataset,
        IPTTS_gender_dataset,
        IPTTS_social_dataset,
    ) = data_loader(
        seed,
        dataset=dataset,
        CDA_examples_ranking=CDA_examples_ranking,
        data_augmentation_ratio=data_augmentation_ratio,
        data_diet_examples_ranking=data_diet_examples_ranking,
        data_diet_factual_ratio=data_diet_factual_ratio,
        data_diet_counterfactual_ratio=data_diet_counterfactual_ratio,
        data_substitution_ratio=data_substitution_ratio,
        max_length=max_length,
        classifier_model=classifier_model,
        IPTTS=True,
    )

    num_labels = len(set(train_dataset.labels))

    if classifier_model in [
        "bert-base-cased",
        "bert-large-cased",
        "distilbert-base-cased",
        "bert-base-uncased",
        "bert-large-uncased",
        "distilbert-base-uncased",
    ]:
        huggingface_model = BertForSequenceClassification
    elif classifier_model in ["roberta-base", "distilroberta-base"]:
        huggingface_model = RobertaForSequenceClassification

    model_checkpoint_path = "./saved_models/cached_models/" + classifier_model
    model_before_bias_reduction = huggingface_model.from_pretrained(
        model_checkpoint_path, num_labels=len(set(val_dataset.labels))
    ).to(device)

    # Load the model that has the best performance on the validation data
    model_before_bias_reduction.load_state_dict(
        torch.load(
            model_dir
            + classifier_model
            + "_"
            + dataset
            + "_"
            + method
            + "_"
            + data_diet_examples_ranking
            + "_"
            + str(data_augmentation_ratio)
            + "_"
            + str(data_diet_factual_ratio)
            + "_"
            + str(data_diet_counterfactual_ratio)
            + "_biased_best.pt",
            map_location=device,
        )
    )

    for split_name, split in zip(
        ["test", "validation", "IPTTS_gender_bias", "IPTTS_social_bias"],
        [test_dataset, val_dataset, IPTTS_gender_dataset, IPTTS_social_dataset],
    ):

        # We compute the metrics that we need by calling this function
        (
            AUC,
            accuracy,
            FPED,
            FNED,
            TPED,
            TNED,
            demographic_parity,
            validation_bias,
            equality_of_odds,
        ) = compute_metrics(
            split,
            split_name,
            dataset,
            model_before_bias_reduction,
            model_after_bias_reduction,
            batch_size_debiased_model,
            num_labels,
            output_dir,
            use_amulet,
            use_wandb,
            method,
            classifier_model,
        )

        if split_name == "IPTTS_gender_bias":
            for metric in [
                FPED,
                FNED,
                TPED,
                TNED,
                demographic_parity,
                equality_of_odds,
            ]:
                all_metrics.append(metric)
        elif split_name == "validation":
            for metric in [AUC, accuracy, validation_bias]:
                all_metrics.append(metric)
        else:
            for metric in [AUC, accuracy]:
                all_metrics.append(metric)

    # We create a directory for saving the metrics, if it doesnt exist
    file_directory = output_dir + dataset + "_" + method + "_" + classifier_model

    Path(file_directory).mkdir(parents=True, exist_ok=True)

    # We now save the metrics in a json file
    output_file = file_directory + "/metrics.json"
    with open(output_file, "w+") as f:
        json.dump(all_metrics, f, indent=2)


def compute_metrics(
    split_dataset,
    split_name,
    dataset_name,
    model_before_bias_reduction,
    model_after_bias_reduction,
    batch_size_debiased_model,
    num_labels,
    output_dir,
    use_amulet,
    use_wandb,
    method,
    classifier_model,
):
    """
    Compute the performance and bias metrics before and after applying our
    de-biasing algorithm on a specific split from the dataset.
    args:
        split_dataset: the dataset split object on which the metrics are measured.
        split_name: the name of the split on which the metrics are computed.
        dataset_name: the name of the dataset used
        model_before_bias_reduction: the model before updating its weights for bias reduction
        model_after_bias_reduction: the model after updating its weights for bias reduction
        batch_size_debiased_model: the size of the batch used
        num_labels: the numnber of labels in the dataset
        output_dir: the directory to the output
        use_amulet: whether or not to run the code on Amulet, which is the cluster used at Microsoft research
        use_wandb: whether or not to use wandb
        method: the debiasing method used
        classifier_model: the model name
    returns:
        the function returns the folowing metrics: accuracy, AUC, FNED, and FPED.
    """
    with torch.no_grad():

        accuracy = {}
        AUC = {}
        FPR = {}
        FNR = {}
        TPR = {}
        TNR = {}
        FPED = {}
        FNED = {}
        TPED = {}
        TNED = {}
        demographic_parity = {}
        validation_bias = {}
        equal_opportunity_y_equal_0 = {}
        equal_opportunity_y_equal_1 = {}
        equality_of_odds = {}

        y_pred_after_bias_reduction = torch.ones([0, num_labels]).to(device)
        y_pred_before_bias_reduction = torch.ones([0, num_labels]).to(device)
        y_pred_gender_swap_after_bias_reduction = torch.ones([0, num_labels]).to(device)

        for i in range(int(np.ceil(len(split_dataset) / batch_size_debiased_model))):

            results_after_bias_reduction = model_after_bias_reduction.forward(
                input_ids=torch.tensor(
                    split_dataset.encodings["input_ids"][
                        i
                        * batch_size_debiased_model : (i + 1)
                        * batch_size_debiased_model
                    ]
                ).to(device),
                attention_mask=torch.tensor(
                    split_dataset.encodings["attention_mask"][
                        i
                        * batch_size_debiased_model : (i + 1)
                        * batch_size_debiased_model
                    ]
                ).to(device),
            )[0]

            results_before_bias_reduction = model_before_bias_reduction.forward(
                input_ids=torch.tensor(
                    split_dataset.encodings["input_ids"][
                        i
                        * batch_size_debiased_model : (i + 1)
                        * batch_size_debiased_model
                    ]
                ).to(device),
                attention_mask=torch.tensor(
                    split_dataset.encodings["attention_mask"][
                        i
                        * batch_size_debiased_model : (i + 1)
                        * batch_size_debiased_model
                    ]
                ).to(device),
            )[0]

            if split_name == "validation":
                # Compute the predictions for the gender-swapped validation examples to determine the fairness hyperparameters a and b
                results_gender_swap_after_bias_reduction = (
                    model_after_bias_reduction.forward(
                        input_ids=torch.tensor(
                            split_dataset.encodings_gender_swap["input_ids"][
                                i
                                * batch_size_debiased_model : (i + 1)
                                * batch_size_debiased_model
                            ]
                        ).to(device),
                        attention_mask=torch.tensor(
                            split_dataset.encodings_gender_swap["attention_mask"][
                                i
                                * batch_size_debiased_model : (i + 1)
                                * batch_size_debiased_model
                            ]
                        ).to(device),
                    )[0]
                )

            # Get the predictions of the new batch
            batch_original_gender = results_after_bias_reduction
            # Add them to the total predictions
            y_pred_after_bias_reduction = torch.cat(
                (y_pred_after_bias_reduction, batch_original_gender), 0
            )

            # Get the predictions of the new batch
            batch_original_gender = results_before_bias_reduction
            # Add them to the total predictions
            y_pred_before_bias_reduction = torch.cat(
                (y_pred_before_bias_reduction, batch_original_gender), 0
            )

            if split_name == "validation":
                # Get the predictions of the new gender-swapped batch
                batch_gender_swap = results_gender_swap_after_bias_reduction
                # Add them to the total predictions
                y_pred_gender_swap_after_bias_reduction = torch.cat(
                    (y_pred_gender_swap_after_bias_reduction, batch_gender_swap), 0
                )
        # ===================================================#
        # Here we calculate the accuracy
        accuracy[split_name + " accuracy before bias reduction"] = (
            torch.sum(
                torch.argmax(y_pred_before_bias_reduction, axis=1).to(device)
                == torch.tensor(split_dataset.labels).to(device)
            )
            / len(split_dataset.labels)
        ).tolist()
        accuracy[split_name + " accuracy after bias reduction"] = (
            torch.sum(
                torch.argmax(y_pred_after_bias_reduction, axis=1).to(device)
                == torch.tensor(split_dataset.labels).to(device)
            )
            / len(split_dataset.labels)
        ).tolist()

        # ===================================================#
        # Here we calculate the AUC score
        if len(set(split_dataset.labels)) > 2:
            # multiclass problems
            AUC[split_name + " AUC before bias reduction"] = roc_auc_score(
                np.array(split_dataset.labels),
                np.array(F.softmax(y_pred_before_bias_reduction, 1).cpu()),
                multi_class="ovo",
            )

            AUC[split_name + " AUC after bias reduction"] = roc_auc_score(
                np.array(split_dataset.labels),
                np.array(F.softmax(y_pred_after_bias_reduction, 1).cpu()),
                multi_class="ovo",
            )

        else:
            AUC[split_name + " AUC before bias reduction"] = roc_auc_score(
                np.array(split_dataset.labels),
                np.array(F.softmax(y_pred_before_bias_reduction, 1).cpu())[:, 1],
            )
            AUC[split_name + " AUC after bias reduction"] = roc_auc_score(
                np.array(split_dataset.labels),
                np.array(F.softmax(y_pred_after_bias_reduction, 1).cpu())[:, 1],
            )

        if split_name == "validation":
            # We compute the bias on the validation dataset to set the hyperparameters, while the IPTTS dataset is to compute the final bias using the best hyperparameters
            validation_bias[split_name + " bias after bias reduction"] = torch.abs(
                torch.sum(
                    torch.argmax(y_pred_after_bias_reduction, axis=1)
                    .double()
                    .to(device)
                )
                - torch.sum(
                    torch.argmax(y_pred_gender_swap_after_bias_reduction, axis=1)
                    .double()
                    .to(device)
                )
            ).tolist()
        #### Log the AUC metric
        logs = dict()

        logs[split_name + " AUC before bias reduction"] = AUC[
            split_name + " AUC before bias reduction"
        ]
        logs[split_name + " AUC after bias reduction"] = AUC[
            split_name + " AUC after bias reduction"
        ]
        if split_name == "validation":
            logs[split_name + " bias after bias reduction"] = validation_bias[
                split_name + " bias after bias reduction"
            ]

        wandb.log(logs)
        # ===================================================#
        # Here we calculate the FPR

        FPR["FPR before bias reduction"] = (
            torch.sum(
                torch.logical_and(
                    torch.argmax(y_pred_before_bias_reduction, axis=1).to(device) == 1,
                    torch.tensor(split_dataset.labels).to(device) == 0,
                )
            )
            / torch.sum(torch.tensor(split_dataset.labels).to(device) == 0)
        ).tolist()
        FPR["FPR after bias reduction"] = (
            torch.sum(
                torch.logical_and(
                    torch.argmax(y_pred_after_bias_reduction, axis=1).to(device) == 1,
                    torch.tensor(split_dataset.labels).to(device) == 0,
                )
            )
            / torch.sum(torch.tensor(split_dataset.labels).to(device) == 0)
        ).tolist()

        # ===================================================#
        # Here we calculate the FNR

        FNR["FNR before bias reduction"] = (
            torch.sum(
                torch.logical_and(
                    torch.argmax(y_pred_before_bias_reduction, axis=1).to(device) == 0,
                    torch.tensor(split_dataset.labels).to(device) == 1,
                )
            )
            / torch.sum(torch.tensor(split_dataset.labels).to(device) == 1)
        ).tolist()
        FNR["FNR after bias reduction"] = (
            torch.sum(
                torch.logical_and(
                    torch.argmax(y_pred_after_bias_reduction, axis=1).to(device) == 0,
                    torch.tensor(split_dataset.labels).to(device) == 1,
                )
            )
            / torch.sum(torch.tensor(split_dataset.labels).to(device) == 1)
        ).tolist()

        # ===================================================#
        # Here we calculate the TPR

        TPR["TPR before bias reduction"] = (
            torch.sum(
                torch.logical_and(
                    torch.argmax(y_pred_before_bias_reduction, axis=1).to(device) == 1,
                    torch.tensor(split_dataset.labels).to(device) == 1,
                )
            )
            / torch.sum(torch.tensor(split_dataset.labels).to(device) == 1)
        ).tolist()
        TPR["TPR after bias reduction"] = (
            torch.sum(
                torch.logical_and(
                    torch.argmax(y_pred_after_bias_reduction, axis=1).to(device) == 1,
                    torch.tensor(split_dataset.labels).to(device) == 1,
                )
            )
            / torch.sum(torch.tensor(split_dataset.labels).to(device) == 1)
        ).tolist()

        # ===================================================#
        # Here we calculate the TNR

        TNR["TNR before bias reduction"] = (
            torch.sum(
                torch.logical_and(
                    torch.argmax(y_pred_before_bias_reduction, axis=1).to(device) == 0,
                    torch.tensor(split_dataset.labels).to(device) == 0,
                )
            )
            / torch.sum(torch.tensor(split_dataset.labels).to(device) == 0)
        ).tolist()
        TNR["TNR after bias reduction"] = (
            torch.sum(
                torch.logical_and(
                    torch.argmax(y_pred_after_bias_reduction, axis=1).to(device) == 0,
                    torch.tensor(split_dataset.labels).to(device) == 0,
                )
            )
            / torch.sum(torch.tensor(split_dataset.labels).to(device) == 0)
        ).tolist()

        # ===================================================#
        # Here we calculate the FNED and FPED metrics, as described in https://arxiv.org/pdf/2004.14088.pdf
        if split_name == "IPTTS_gender_bias":
            data_IPTTS = pd.read_csv("./data/" + "madlib.csv")
            idxs_male = [
                i
                for i in range(len(data_IPTTS["gender"].values))
                if data_IPTTS["gender"].values[i] == "male"
            ]
            idxs_female = [
                i
                for i in range(len(data_IPTTS["gender"].values))
                if data_IPTTS["gender"].values[i] == "female"
            ]

        elif split_name == "IPTTS_social_bias":
            data_IPTTS = pd.read_csv("./data/" + "bias_madlibs_77k.csv")

        if split_name in [
            "IPTTS_gender_bias",
            "IPTTS_social_bias",
        ]:
            # Saving the model's predcitions before and after debiasing, on the
            # IPTTs dataset.
            data_IPTTS["predcition before debiasing"] = list(
                F.softmax(y_pred_before_bias_reduction, 1)[:, 1].cpu().detach().numpy()
            )
            data_IPTTS["prediction after debiasing"] = list(
                F.softmax(y_pred_after_bias_reduction, 1)[:, 1].cpu().detach().numpy()
            )
            data_IPTTS["number of tokens"] = data_IPTTS[data_IPTTS.columns[0]].apply(
                lambda x: len(re.findall(r"\w+", x))
            )
            # We create a directory for saving the analysis file, if it doesnt exist
            file_directory = output_dir + "analysis/"

            Path(file_directory).mkdir(parents=True, exist_ok=True)

            data_IPTTS.to_csv(
                file_directory
                + split_name
                + "_analysis"
                + "_"
                + dataset_name
                + "_"
                + method
                + "_"
                + classifier_model
                + ".csv",
                index=False,
            )

        if split_name == "IPTTS_gender_bias":
            # We just initialize the FNED and FPED to zeros, before we compute them.
            FPED["FPED before bias reduction"], FPED["FPED after bias reduction"] = 0, 0
            FNED["FNED before bias reduction"], FNED["FNED after bias reduction"] = 0, 0
            TPED["TPED before bias reduction"], TPED["TPED after bias reduction"] = 0, 0
            TNED["TNED before bias reduction"], TNED["TNED after bias reduction"] = 0, 0
            equal_opportunity_y_equal_0["before bias reduction"] = torch.tensor(0).to(
                device
            )
            equal_opportunity_y_equal_0["after bias reduction"] = torch.tensor(0).to(
                device
            )
            equal_opportunity_y_equal_1["before bias reduction"] = torch.tensor(0).to(
                device
            )
            equal_opportunity_y_equal_1["after bias reduction"] = torch.tensor(0).to(
                device
            )
            num_positive_examples = torch.sum(
                torch.tensor(split_dataset.labels).to(device) == 1
            )
            num_negative_examples = torch.sum(
                torch.tensor(split_dataset.labels).to(device) == 0
            )
            for idx in [idxs_male, idxs_female]:
                num_positive_examples = torch.sum(
                    torch.tensor(split_dataset.labels).to(device)[idx] == 1
                )
                num_negative_examples = torch.sum(
                    torch.tensor(split_dataset.labels).to(device)[idx] == 0
                )
                pred_before_bias_reduction = torch.argmax(
                    y_pred_before_bias_reduction, axis=1
                ).to(device)[idx]
                pred_after_bias_reduction = torch.argmax(
                    y_pred_after_bias_reduction, axis=1
                ).to(device)[idx]
                ground_truth = torch.tensor(split_dataset.labels).to(device)[idx]

                FPED["FPED before bias reduction"] += torch.abs(
                    FPR["FPR before bias reduction"]
                    - (
                        torch.sum(
                            torch.logical_and(
                                pred_before_bias_reduction == 1, ground_truth == 0
                            )
                        )
                        / num_negative_examples
                    )
                ).tolist()
                FPED["FPED after bias reduction"] += torch.abs(
                    FPR["FPR after bias reduction"]
                    - (
                        torch.sum(
                            torch.logical_and(
                                pred_after_bias_reduction == 1, ground_truth == 0
                            )
                        )
                        / num_negative_examples
                    )
                ).tolist()

                FNED["FNED before bias reduction"] += torch.abs(
                    FNR["FNR before bias reduction"]
                    - (
                        torch.sum(
                            torch.logical_and(
                                pred_before_bias_reduction == 0, ground_truth == 1
                            )
                        )
                        / num_positive_examples
                    )
                ).tolist()
                FNED["FNED after bias reduction"] += torch.abs(
                    FNR["FNR after bias reduction"]
                    - (
                        torch.sum(
                            torch.logical_and(
                                pred_after_bias_reduction == 0, ground_truth == 1
                            )
                        )
                        / num_positive_examples
                    )
                ).tolist()

                TPED["TPED before bias reduction"] += torch.abs(
                    TPR["TPR before bias reduction"]
                    - (
                        torch.sum(
                            torch.logical_and(
                                pred_before_bias_reduction == 1, ground_truth == 1
                            )
                        )
                        / num_positive_examples
                    )
                ).tolist()
                TPED["TPED after bias reduction"] += torch.abs(
                    TPR["TPR after bias reduction"]
                    - (
                        torch.sum(
                            torch.logical_and(
                                pred_after_bias_reduction == 1, ground_truth == 1
                            )
                        )
                        / num_positive_examples
                    )
                ).tolist()

                TNED["TNED before bias reduction"] += torch.abs(
                    TNR["TNR before bias reduction"]
                    - (
                        torch.sum(
                            torch.logical_and(
                                pred_before_bias_reduction == 0, ground_truth == 0
                            )
                        )
                        / num_negative_examples
                    )
                ).tolist()
                TNED["TNED after bias reduction"] += torch.abs(
                    TNR["TNR after bias reduction"]
                    - (
                        torch.sum(
                            torch.logical_and(
                                pred_after_bias_reduction == 0, ground_truth == 0
                            )
                        )
                        / num_negative_examples
                    )
                ).tolist()

                equal_opportunity_y_equal_0["before bias reduction"] = torch.abs(
                    equal_opportunity_y_equal_0["before bias reduction"]
                ) - (
                    torch.sum(
                        torch.logical_and(
                            pred_before_bias_reduction == 1, ground_truth == 0
                        )
                    )
                    / num_negative_examples
                )
                equal_opportunity_y_equal_1["before bias reduction"] = torch.abs(
                    equal_opportunity_y_equal_1["before bias reduction"]
                ) - (
                    torch.sum(
                        torch.logical_and(
                            pred_before_bias_reduction == 1, ground_truth == 1
                        )
                    )
                    / num_positive_examples
                )

                equal_opportunity_y_equal_0["after bias reduction"] = torch.abs(
                    equal_opportunity_y_equal_0["after bias reduction"]
                ) - (
                    torch.sum(
                        torch.logical_and(
                            pred_after_bias_reduction == 1, ground_truth == 0
                        )
                    )
                    / num_negative_examples
                )
                equal_opportunity_y_equal_1["after bias reduction"] = torch.abs(
                    equal_opportunity_y_equal_1["after bias reduction"]
                ) - (
                    torch.sum(
                        torch.logical_and(
                            pred_after_bias_reduction == 1, ground_truth == 1
                        )
                    )
                    / num_positive_examples
                )

            demographic_parity["demographic parity before bias reduction"] = (
                1
                - torch.abs(
                    torch.mean(
                        torch.argmax(y_pred_before_bias_reduction, axis=1)
                        .double()
                        .to(device)[idxs_male]
                    )
                    - torch.mean(
                        torch.argmax(y_pred_before_bias_reduction, axis=1)
                        .double()
                        .to(device)[idxs_female]
                    )
                ).tolist()
            )

            demographic_parity["demographic parity after bias reduction"] = (
                1
                - torch.abs(
                    torch.mean(
                        torch.argmax(y_pred_after_bias_reduction, axis=1)
                        .double()
                        .to(device)[idxs_male]
                    )
                    - torch.mean(
                        torch.argmax(y_pred_after_bias_reduction, axis=1)
                        .double()
                        .to(device)[idxs_female]
                    )
                ).tolist()
            )

            equal_opportunity_y_equal_0["before bias reduction"] = 1 - torch.abs(
                equal_opportunity_y_equal_0["before bias reduction"]
            )
            equal_opportunity_y_equal_1["before bias reduction"] = 1 - torch.abs(
                equal_opportunity_y_equal_1["before bias reduction"]
            )

            equal_opportunity_y_equal_0["after bias reduction"] = 1 - torch.abs(
                equal_opportunity_y_equal_0["after bias reduction"]
            )
            equal_opportunity_y_equal_1["after bias reduction"] = 1 - torch.abs(
                equal_opportunity_y_equal_1["after bias reduction"]
            )

            equality_of_odds["equality of odds before bias reduction"] = (
                0.5
                * (
                    equal_opportunity_y_equal_0["before bias reduction"]
                    + equal_opportunity_y_equal_1["before bias reduction"]
                ).tolist()
            )
            equality_of_odds["equality of odds after bias reduction"] = (
                0.5
                * (
                    equal_opportunity_y_equal_0["after bias reduction"]
                    + equal_opportunity_y_equal_1["after bias reduction"]
                ).tolist()
            )

            logs = dict()

            #### Log the metrics
            logs["Demographic parity before bias reduction"] = demographic_parity[
                "demographic parity before bias reduction"
            ]
            logs["Demographic parity after bias reduction"] = demographic_parity[
                "demographic parity after bias reduction"
            ]

            wandb.log(logs)

        return (
            AUC,
            accuracy,
            FPED,
            FNED,
            TPED,
            TNED,
            demographic_parity,
            validation_bias,
            equality_of_odds,
        )
