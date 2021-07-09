import pandas as pd
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
import torch
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softmax = torch.nn.Softmax(dim=1).to(device)


def measure_bias_metrics(model, tokenizer, args):
    """
    Compute metrics that measure the bias before and after applying our de-biasing algorithm.
    The metrics that we compute are the following:
    1) Demagraphic parity
    2) Equality of odds
    3) Counterfactual token fairness
    4) True negative rate
    5) True positive rate
    6) Equality of opportunity
    args:
        args: the arguments given by the user
        model: the model after updating its weights based on the policy gradient algorithm.
        tokenizer: the tokenizer used before giving the sentences to the classifier
    returns:
        the function doesnt return anything, since all the metrics are saved in txt files.
    """
    demographic_parity = {}
    CTF = {}
    # Load test data
    test_data = pd.read_csv("./data/" + args.dataset + "_valid_original_gender.csv")
    input_column_name = test_data.columns[1]
    label_column_name = test_data.columns[2]
    number_of_labels = len(test_data.Class.unique())
    X_test = list(test_data[input_column_name])
    X_test_tokenized = tokenizer(
        X_test, padding=True, truncation=True, max_length=args.max_length
    )

    test_data_opposite_gender = pd.read_csv(
        "./data/" + args.dataset + "_valid_gender_swap.csv"
    )
    input_column_name_gender_swap = test_data_opposite_gender.columns[1]
    X_test_opposite_gender = list(
        test_data_opposite_gender[input_column_name_gender_swap]
    )
    X_test_tokenized_opposite_gender = tokenizer(
        X_test_opposite_gender,
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )


    # Create torch dataset
    test_dataset = Dataset(X_test_tokenized)
    # Define test trainer
    test_trainer = Trainer(model)
    # Make prediction
    raw_pred, _, _ = test_trainer.predict(test_dataset)
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred[0], axis=1)

    # Load test data for the opposite gender
    # Create torch dataset
    test_dataset_opposite_gender = Dataset(X_test_tokenized_opposite_gender)
    # Make prediction
    raw_pred_opposite_gender, _, _ = test_trainer.predict(test_dataset_opposite_gender)
    # Preprocess raw predictions
    y_pred_opposite_gender = np.argmax(raw_pred_opposite_gender[0], axis=1)

    demographic_parity["after_bias_reduction"] = 1 - torch.abs(
        torch.mean(torch.from_numpy(y_pred).double())
        - torch.mean(torch.from_numpy(y_pred_opposite_gender).double())
    )
    CTF["after_bias_reduction"] = torch.mean(
        torch.abs(
            softmax(torch.from_numpy(raw_pred[0]))[:, 1]
            - softmax(torch.from_numpy(raw_pred_opposite_gender[0]))[:, 1]
        )
    )

    # We also compute the same metric before reducing the bias
    # Load trained model
    model_path = "./saved_models/checkpoint-500"
    model_before_bias_reduction = BertForSequenceClassification.from_pretrained(
        model_path, num_labels=number_of_labels, output_attentions=True
    )

    # Define test trainer
    test_trainer_before_bias_reduction = Trainer(model_before_bias_reduction)
    # Make prediction
    raw_pred, _, _ = test_trainer_before_bias_reduction.predict(test_dataset)
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred[0], axis=1)

    # Load test data for the opposite gender
    # Create torch dataset
    test_dataset_opposite_gender = Dataset(X_test_tokenized_opposite_gender)
    # Make prediction
    raw_pred_opposite_gender, _, _ = test_trainer_before_bias_reduction.predict(
        test_dataset_opposite_gender
    )
    # Preprocess raw predictions
    y_pred_opposite_gender = np.argmax(raw_pred_opposite_gender[0], axis=1)

    demographic_parity["before_bias_reduction"] = 1 - torch.abs(
        torch.mean(torch.from_numpy(y_pred).double())
        - torch.mean(torch.from_numpy(y_pred_opposite_gender).double())
    )
    CTF["before_bias_reduction"] = torch.mean(
        torch.abs(
            softmax(torch.from_numpy(raw_pred[0]))[:, 1]
            - softmax(torch.from_numpy(raw_pred_opposite_gender[0]))[:, 1]
        )
    )

    output_file = "./output/demographic_parity.txt"
    f = open(output_file, "w")
    f.write(str(demographic_parity))
    f.close()

    output_file = "./output/CTF.txt"
    f = open(output_file, "w")
    f.write(str(CTF))
    f.close()
    # ===================================================#

    equality_of_opportunity_y_equal_0 = {}
    TNR = {}
    # Load test data
    test_data = pd.read_csv("./data/" + args.dataset + "_valid_original_gender.csv")
    test_data_non_sexist_tweets = test_data.loc[test_data[label_column_name] == 0]
    X_test = list(test_data_non_sexist_tweets[input_column_name])
    X_test_tokenized = tokenizer(
        X_test, padding=True, truncation=True, max_length=args.max_length
    )

    test_data_opposite_gender = pd.read_csv(
        "./data/" + args.dataset + "_valid_gender_swap.csv"
    )
    test_data_opposite_gender_non_sexist_tweets = test_data_opposite_gender.loc[
        test_data_opposite_gender[label_column_name] == 0
    ]
    X_test_opposite_gender = list(
        test_data_opposite_gender_non_sexist_tweets[input_column_name_gender_swap]
    )
    X_test_tokenized_opposite_gender = tokenizer(
        X_test_opposite_gender,
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )

    # Create torch dataset
    test_dataset = Dataset(X_test_tokenized)
    # Define test trainer
    test_trainer = Trainer(model)
    # Make prediction
    raw_pred, _, _ = test_trainer.predict(test_dataset)
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred[0], axis=1)

    # Load test data for the opposite gender
    # Create torch dataset
    test_dataset_opposite_gender = Dataset(X_test_tokenized_opposite_gender)
    # Make prediction
    raw_pred_opposite_gender, _, _ = test_trainer.predict(test_dataset_opposite_gender)
    # Preprocess raw predictions
    y_pred_opposite_gender = np.argmax(raw_pred_opposite_gender[0], axis=1)

    equality_of_opportunity_y_equal_0["after_bias_reduction"] = 1 - torch.abs(
        torch.mean(torch.from_numpy(y_pred).double())
        - torch.mean(torch.from_numpy(y_pred_opposite_gender).double())
    )
    TNR["after_bias_reduction"] = 1 - torch.mean(torch.from_numpy(y_pred).double())

    # Make prediction
    raw_pred, _, _ = test_trainer_before_bias_reduction.predict(test_dataset)
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred[0], axis=1)

    # Load test data for the opposite gender
    # Create torch dataset
    test_dataset_opposite_gender = Dataset(X_test_tokenized_opposite_gender)
    # Make prediction
    raw_pred_opposite_gender, _, _ = test_trainer_before_bias_reduction.predict(
        test_dataset_opposite_gender
    )
    # Preprocess raw predictions
    y_pred_opposite_gender = np.argmax(raw_pred_opposite_gender[0], axis=1)

    equality_of_opportunity_y_equal_0["before_bias_reduction"] = 1 - torch.abs(
        torch.mean(torch.from_numpy(y_pred).double())
        - torch.mean(torch.from_numpy(y_pred_opposite_gender).double())
    )
    TNR["before_bias_reduction"] = 1 - torch.mean(torch.from_numpy(y_pred).double())

    output_file = "./output/equality_of_opportunity_y_equal_0.txt"
    f = open(output_file, "w")
    f.write(str(equality_of_opportunity_y_equal_0))
    f.close()

    output_file = "./output/TNR.txt"
    f = open(output_file, "w")
    f.write(str(TNR))
    f.close()

    # ===================================================#

    equality_of_opportunity_y_equal_1 = {}
    TPR = {}
    # Load test data
    test_data = pd.read_csv("./data/" + args.dataset + "_valid_original_gender.csv")
    test_data_sexist_tweets = test_data.loc[test_data[label_column_name] == 1]
    X_test = list(test_data_sexist_tweets[input_column_name])
    X_test_tokenized = tokenizer(
        X_test, padding=True, truncation=True, max_length=args.max_length
    )

    test_data_opposite_gender = pd.read_csv(
        "./data/" + args.dataset + "_valid_gender_swap.csv"
    )
    test_data_opposite_gender_sexist_tweets = test_data_opposite_gender.loc[
        test_data_opposite_gender[label_column_name] == 1
    ]
    X_test_opposite_gender = list(
        test_data_opposite_gender_sexist_tweets[input_column_name_gender_swap]
    )
    X_test_tokenized_opposite_gender = tokenizer(
        X_test_opposite_gender,
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )

    # Create torch dataset
    test_dataset = Dataset(X_test_tokenized)
    # Define test trainer
    test_trainer = Trainer(model)
    # Make prediction
    raw_pred, _, _ = test_trainer.predict(test_dataset)
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred[0], axis=1)

    # Load test data for the opposite gender
    # Create torch dataset
    test_dataset_opposite_gender = Dataset(X_test_tokenized_opposite_gender)
    # Make prediction
    raw_pred_opposite_gender, _, _ = test_trainer.predict(test_dataset_opposite_gender)
    # Preprocess raw predictions
    y_pred_opposite_gender = np.argmax(raw_pred_opposite_gender[0], axis=1)

    equality_of_opportunity_y_equal_1["after_bias_reduction"] = 1 - torch.abs(
        torch.mean(torch.from_numpy(y_pred).double())
        - torch.mean(torch.from_numpy(y_pred_opposite_gender).double())
    )
    TPR["after_bias_reduction"] = torch.mean(torch.from_numpy(y_pred).double())

    # Make prediction
    raw_pred, _, _ = test_trainer_before_bias_reduction.predict(test_dataset)
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred[0], axis=1)

    # Load test data for the opposite gender
    # Create torch dataset
    test_dataset_opposite_gender = Dataset(X_test_tokenized_opposite_gender)
    # Make prediction
    raw_pred_opposite_gender, _, _ = test_trainer_before_bias_reduction.predict(
        test_dataset_opposite_gender
    )
    # Preprocess raw predictions
    y_pred_opposite_gender = np.argmax(raw_pred_opposite_gender[0], axis=1)

    equality_of_opportunity_y_equal_1["before_bias_reduction"] = 1 - torch.abs(
        torch.mean(torch.from_numpy(y_pred).double())
        - torch.mean(torch.from_numpy(y_pred_opposite_gender).double())
    )
    TPR["before_bias_reduction"] = torch.mean(torch.from_numpy(y_pred).double())

    output_file = "./output/equality_of_opportunity_y_equal_1.txt"
    f = open(output_file, "w")
    f.write(str(equality_of_opportunity_y_equal_1))
    f.close()

    output_file = "./output/TPR.txt"
    f = open(output_file, "w")
    f.write(str(TPR))
    f.close()
    # ===================================================#

    equality_of_odds = {}
    equality_of_odds["after_bias_reduction"] = 0.5 * (
        equality_of_opportunity_y_equal_0["after_bias_reduction"]
        + equality_of_opportunity_y_equal_1["after_bias_reduction"]
    )
    equality_of_odds["before_bias_reduction"] = 0.5 * (
        equality_of_opportunity_y_equal_0["before_bias_reduction"]
        + equality_of_opportunity_y_equal_1["before_bias_reduction"]
    )
    output_file = "./output/equality_of_odds.txt"
    f = open(output_file, "w")
    f.write(str(equality_of_odds))
    f.close()

    # ===================================================#


# Some of the following parts are taken from https://towardsdatascience.com/fine-tuning-pretrained-nlp-models-with-huggingfaces-trainer-6326a4456e7b by Vincent Tan
def train_classifier(args):
    """
    Train a classifier to be used as our starting point for polcy gradient. We can either train from scratch or load a pretrained model depending on the user's choice.
    args:
        args: the arguments given by the user
    returns:
        model: the model that is going to be our starting point for policy gradient
        tokenizer: the tokenizer used before giving the sentences to the classifier model
    """
    # Read data
    data_train = pd.read_csv("./data/" + args.dataset + "_train_original_gender.csv")
    data_valid = pd.read_csv("./data/" + args.dataset + "_valid_original_gender.csv")

    if args.load_pretrained_classifier:

        tokenizer = BertTokenizer.from_pretrained(args.classifier_model)
        model_path = "./saved_models/checkpoint-500"
        model = BertForSequenceClassification.from_pretrained(
            model_path, num_labels=len(data.Class.unique()), output_attentions=True
        )

    else:
        # Define pretrained tokenizer and model
        model_name = args.classifier_model
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(data_train.Class.unique()),
            output_attentions=True,
        )

        # ----- 1. Preprocess data -----#
        # Preprocess data
        X_train = list(data_train[data_train.columns[1]])
        y_train = list(data_train[data_train.columns[2]])
        X_val = list(data_valid[data_valid.columns[1]])
        y_val = list(data_valid[data_valid.columns[2]])
        X_train_tokenized = tokenizer(
            X_train, padding=True, truncation=True, max_length=args.max_length
        )
        X_val_tokenized = tokenizer(
            X_val, padding=True, truncation=True, max_length=args.max_length
        )

        train_dataset = Dataset(X_train_tokenized, y_train)
        val_dataset = Dataset(X_val_tokenized, y_val)

        # Define Trainer parameters

        # Define Trainer
        classifier_args = TrainingArguments(
            output_dir="saved_models",
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=3000,
            per_device_train_batch_size=args.batch_size_classifier,
            per_device_eval_batch_size=args.batch_size_classifier,
            num_train_epochs=args.num_epochs_classifier,
            load_best_model_at_end=True,
        )
        trainer = Trainer(
            model=model,
            args=classifier_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Train pre-trained model
        trainer.train()
    return model, tokenizer


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred[0], axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average="micro")
    precision = precision_score(y_true=labels, y_pred=pred, average="micro")
    f1 = f1_score(y_true=labels, y_pred=pred, average="micro")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
