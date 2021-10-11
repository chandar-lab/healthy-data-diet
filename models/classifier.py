from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from models.data_loader import data_loader
from sklearn.metrics import roc_auc_score
import torch
import json
import numpy as np
import torch.nn.functional as F
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assess_performance_and_bias(model_after_bias_reduction, args, run):
    """
    Measure the performance before and after applying our de-biasing algorithm.
    This is done by computing the metrics on the validation, test and IPTTS datasets.
    The IPTTS dataset is a synthetic dataset, developed to measure the bias in the model.
    The metrics that we compute are the following:
    1) AUC
    2) Accuracy
    3) FRED
    4) FPED
    args:
        model_after_bias_reduction: the model after updating its weights due to bias reduction
        args: the arguments given by the user
        run : the index of the current run, that goes from 0 to the number of runs defined by the user
    returns:
        the function doesnt return anything, since all the metrics are saved in json files.
    """
    # We need to load the datasets on which we measure the metrics.
    if (args.method == "baseline_data_augmentation" == True):
      train_dataset, val_dataset, test_dataset, IPTTS_dataset = data_loader(args, IPTTS=True, apply_data_augmentation=True)
    else:
      train_dataset, val_dataset, test_dataset, IPTTS_dataset = data_loader(args, IPTTS=True, apply_data_augmentation=False)
    checkpoint_steps = (
        int(train_dataset.__len__() / args.batch_size_pretraining)
        * args.num_epochs_pretraining
    )

    # Load trained model before bias reduction
    model_path = "./saved_models/checkpoint-" + str(checkpoint_steps)
    model_before_bias_reduction = BertForSequenceClassification.from_pretrained(
        model_path, num_labels=len(set(val_dataset.labels)), output_attentions=args.analyze_results
    ).to(device)

    for split_name, split in zip(
        ["test", "validation", "IPTTS"], [val_dataset, test_dataset, IPTTS_dataset]
    ):

        # We compute the metrics that we need by calling this function
        AUC, accuracy, FPED, FNED = compute_metrics(
            split,
            split_name,
            args.dataset,
            model_before_bias_reduction,
            model_after_bias_reduction,
            args.batch_size,
        )

        # We now save the metrics in json files
        output_file = (
            "./output/"
            + split_name
            + "_AUC_"
            + args.method
            + "_"
            + args.approach
            + "_"
            + args.dataset
            + "_"
            + args.norm
            + ".json"
        )
        with open(output_file, "w+") as f:
            json.dump(AUC, f, indent=2)

        output_file = (
            "./output/"
            + split_name
            + "_accuracy_"
            + args.method
            + "_"
            + args.approach
            + "_"
            + args.dataset
            + "_"
            + args.norm
            + ".json"
        )
        with open(output_file, "w+") as f:
            json.dump(accuracy, f, indent=2)

        if split_name == "IPTTS":
            # If the dataset is the IPTTS dataset, we also save the FPED and FNED.
            output_file = (
                "./output/"
                + split_name
                + "_FPED_"
                + args.method
                + "_"
                + args.approach
                + "_"
                + args.dataset
                + "_"
                + args.norm
                + ".json"
            )
            with open(output_file, "w+") as f:
                json.dump(FPED, f, indent=2)

            output_file = (
                "./output/"
                + split_name
                + "_FNED_"
                + args.method
                + "_"
                + args.approach
                + "_"
                + args.dataset
                + "_"
                + args.norm
                + ".json"
            )
            with open(output_file, "w+") as f:
                json.dump(FNED, f, indent=2)


def compute_metrics(
    split_dataset,
    split_name,
    dataset_name,
    model_before_bias_reduction,
    model_after_bias_reduction,
    batch_size,
):
    """
    Compute the performance and bias metrics before and after applying our
    de-biasing algorithm on a specific split from the dataset.
    args:
        split_dataset: the dataset split object on which the metrics are measured.
        split_name: the name of the split on which the metrics are computed.
        dataset_name: the nam of the dataset used
        model_before_bias_reduction: the model before updating its weights for bias reduction
        model_after_bias_reduction: the model after updating its weights for bias reduction
        batch_size: the size of the batch used
    returns:
        the function returns the folowing metrics: accuracy, AUC, FNED, and FPED.
    """
    with torch.no_grad():

        accuracy = {}
        AUC = {}
        FPR = {}
        FNR = {}
        FPED = {}
        FNED = {}

        num_labels = len(set(split_dataset.labels))
        y_pred_after_bias_reduction = torch.ones([0, num_labels]).to(device)
        y_pred_before_bias_reduction = torch.ones([0, num_labels]).to(device)

        for i in range(int(np.ceil(len(split_dataset) / batch_size))):

            results_after_bias_reduction = model_after_bias_reduction.forward(
                input_ids=torch.tensor(
                    split_dataset.encodings["input_ids"][
                        i * batch_size : (i + 1) * batch_size
                    ]
                ).to(device),
                attention_mask=torch.tensor(
                    split_dataset.encodings["attention_mask"][
                        i * batch_size : (i + 1) * batch_size
                    ]
                ).to(device),
                token_type_ids=torch.tensor(
                    split_dataset.encodings["token_type_ids"][
                        i * batch_size : (i + 1) * batch_size
                    ]
                ).to(device),
            )[0]

            results_before_bias_reduction = model_before_bias_reduction.forward(
                input_ids=torch.tensor(
                    split_dataset.encodings["input_ids"][
                        i * batch_size : (i + 1) * batch_size
                    ]
                ).to(device),
                attention_mask=torch.tensor(
                    split_dataset.encodings["attention_mask"][
                        i * batch_size : (i + 1) * batch_size
                    ]
                ).to(device),
                token_type_ids=torch.tensor(
                    split_dataset.encodings["token_type_ids"][
                        i * batch_size : (i + 1) * batch_size
                    ]
                ).to(device),
            )[0]

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

        # ===================================================#
        # Here we calculate the accuracy

        accuracy["before_bias_reduction"] = (
            torch.sum(
                torch.argmax(y_pred_before_bias_reduction, axis=1).to(device)
                == torch.tensor(split_dataset.labels).to(device)
            )
            / len(split_dataset.labels)
        ).tolist()
        accuracy["after_bias_reduction"] = (
            torch.sum(
                torch.argmax(y_pred_after_bias_reduction, axis=1).to(device)
                == torch.tensor(split_dataset.labels).to(device)
            )
            / len(split_dataset.labels)
        ).tolist()

        # ===================================================#
        # Here we calculate the AUC score
        AUC["before_bias_reduction"] = roc_auc_score(
            np.array(split_dataset.labels),
            np.array(F.softmax(y_pred_before_bias_reduction, 1).cpu())[:, 1],
        )
        AUC["after_bias_reduction"] = roc_auc_score(
            np.array(split_dataset.labels),
            np.array(F.softmax(y_pred_after_bias_reduction, 1).cpu())[:, 1],
        )

        # ===================================================#
        # Here we calculate the FNR

        FPR["before_bias_reduction"] = (
            torch.sum(
                torch.logical_and(
                    torch.argmax(y_pred_before_bias_reduction, axis=1).to(device) == 1,
                    torch.tensor(split_dataset.labels).to(device) == 0,
                )
            )
            / torch.sum(torch.tensor(split_dataset.labels).to(device) == 0)
        ).tolist()
        FPR["after_bias_reduction"] = (
            torch.sum(
                torch.logical_and(
                    torch.argmax(y_pred_after_bias_reduction, axis=1).to(device) == 1,
                    torch.tensor(split_dataset.labels).to(device) == 0,
                )
            )
            / torch.sum(torch.tensor(split_dataset.labels).to(device) == 0)
        ).tolist()

        # ===================================================#
        # Here we calculate the FPR

        FNR["before_bias_reduction"] = (
            torch.sum(
                torch.logical_and(
                    torch.argmax(y_pred_before_bias_reduction, axis=1).to(device) == 0,
                    torch.tensor(split_dataset.labels).to(device) == 1,
                )
            )
            / torch.sum(torch.tensor(split_dataset.labels).to(device) == 1)
        ).tolist()
        FNR["after_bias_reduction"] = (
            torch.sum(
                torch.logical_and(
                    torch.argmax(y_pred_after_bias_reduction, axis=1).to(device) == 0,
                    torch.tensor(split_dataset.labels).to(device) == 1,
                )
            )
            / torch.sum(torch.tensor(split_dataset.labels).to(device) == 1)
        ).tolist()

        # ===================================================#
        # Here we calculate the FNED and FPED
        if split_name == "IPTTS":
            if(dataset_name == "Twitter_sexism_dataset"):
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
            elif (dataset_name == "Wikipedia_toxicity_dataset" or dataset_name == "Jigsaw_toxicity_dataset"):
                data_IPTTS = pd.read_csv("./data/" + "bias_madlibs_77k.csv")
                idxs_male = list(data_IPTTS[data_IPTTS["text"].str.contains(" male")].index.values)
                idxs_female = list(data_IPTTS[data_IPTTS["text"].str.contains(" female")].index.values)                
            # We just initialize the FNED and FPED to zeros, before we compute them.
            FPED["before_bias_reduction"], FPED["after_bias_reduction"] = 0, 0
            FNED["before_bias_reduction"], FNED["after_bias_reduction"] = 0, 0

            for idx in [idxs_male, idxs_female]:
                num_positive_examples = torch.sum(torch.tensor(split_dataset.labels).to(device)[idx] == 1)
                num_negative_examples = torch.sum(torch.tensor(split_dataset.labels).to(device)[idx] == 0)
                pred_before_bias_reduction = torch.argmax(y_pred_before_bias_reduction,axis = 1).to(device)[idx]
                pred_after_bias_reduction = torch.argmax(y_pred_after_bias_reduction,axis = 1).to(device)[idx]
                ground_truth = torch.tensor(split_dataset.labels).to(device)[idx]
                FPED["before_bias_reduction"] += torch.abs(FPR["before_bias_reduction"]-(torch.sum(torch.logical_and(pred_before_bias_reduction == 1, ground_truth == 0))/num_negative_examples)).tolist()
                FPED["after_bias_reduction"] += torch.abs(FPR["after_bias_reduction"]-(torch.sum(torch.logical_and(pred_after_bias_reduction == 1, ground_truth == 0))/num_negative_examples)).tolist()
  
                FNED["before_bias_reduction"] += torch.abs(FNR["before_bias_reduction"]-(torch.sum(torch.logical_and(pred_before_bias_reduction == 0, ground_truth == 1))/num_positive_examples)).tolist()
                FNED["after_bias_reduction"] += torch.abs(FNR["after_bias_reduction"]-(torch.sum(torch.logical_and(pred_after_bias_reduction == 0, ground_truth == 1))/num_positive_examples)).tolist()
    
        return AUC, accuracy, FPED, FNED


# Some of the following parts are taken from
# https://towardsdatascience.com/fine-tuning-pretrained-nlp-models-with-huggingfaces-trainer-6326a4456e7b
# by Vincent Tan
def train_classifier(args, data_augmentation_flag=None):
    """
    Train a classifier to be used as our starting point for polcy gradient.
    We can either train from scratch or load a pretrained model depending on
    the user's choice.
    args:
        args: the arguments given by the user
        data_augmentation_flag: a flag to choose whether or not to apply data
        augmentation, meaning that the number of examples doubles because we
        flip the gender in each example and add it as a new example.
    returns:
        model: the model that is going to be our starting point for polic
        y gradient
        tokenizer: the tokenizer used before giving the sentences to the
        classifier model
    """
    # Load the dataset
    train_dataset, val_dataset, test_dataset = data_loader(
        args, apply_data_augmentation=data_augmentation_flag
    )
    # The number of epochs afterwhich we save the model. We set it to this
    # value to only save the last model.
    checkpoint_steps = (
        int(train_dataset.__len__() / args.batch_size_pretraining)
        * args.num_epochs_pretraining
    )

    if args.load_pretrained_classifier:

        tokenizer = BertTokenizer.from_pretrained(args.classifier_model)
        model = BertForSequenceClassification.from_pretrained(
            args.model_path + str(checkpoint_steps),
            num_labels=len(set(train_dataset.labels)),
            # We only need the attention weights if we are going to analyze the results
            output_attentions=args.analyze_results,
        )

    else:
        # Define pretrained tokenizer and model
        model_name = args.classifier_model
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(set(train_dataset.labels)),
            # We only need the attention weights if we are going to analyze the results
            output_attentions=args.analyze_results,
        )

        # Define Trainer parameters

        # Define Trainer
        classifier_args = TrainingArguments(
            output_dir=args.output_dir,
            evaluation_strategy="steps",
            eval_steps=checkpoint_steps,
            per_device_train_batch_size=args.batch_size_pretraining,
            per_device_eval_batch_size=args.batch_size_pretraining,
            num_train_epochs=args.num_epochs_pretraining,
            load_best_model_at_end=True,
        )
        trainer = Trainer(
            model=model,
            args=classifier_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Train pre-trained model
        trainer.train()
    return model, tokenizer
