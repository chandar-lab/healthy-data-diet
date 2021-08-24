import pandas as pd
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
import torch
import json
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from models.data_loader import data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softmax = torch.nn.Softmax(dim=1).to(device)


def measure_bias_metrics(model, tokenizer, args):
    """
    Compute metrics that measure the bias before and after applying our
    de-biasing algorithm.
    The metrics that we compute are the following:
    1) Demagraphic parity
    2) Equality of odds
    3) Counterfactual token fairness
    4) True negative rate
    5) True positive rate
    6) Accuracy for both the original and opposite gender (each separately and
    combined)
    7) Equality of opportunity
    args:
        args: the arguments given by the user
        model: the model after updating its weights based on the policy gradient
        algorithm.
        tokenizer: the tokenizer used before giving the sentences to the classifier
    returns:
        the function doesnt return anything, since all the metrics are saved in
        json files.
    """
    demographic_parity = {}
    CTF = {}
    # Load validation data
    valid_data = pd.read_csv("./data/" + args.dataset + "_valid_original_gender.csv")

    input_column_name = valid_data.columns[0]
    label_column_name = valid_data.columns[1]
    number_of_labels = len(valid_data[valid_data.columns[1]].unique())
    X_valid = list(valid_data[input_column_name])
    Y_valid = list(valid_data[label_column_name])
    X_valid_tokenized = tokenizer(
        X_valid, padding=True, truncation=True, max_length=args.max_length
    )

    valid_data_opposite_gender = pd.read_csv(
        "./data/" + args.dataset + "_valid_gender_swap.csv"
    )

    input_column_name_gender_swap = valid_data_opposite_gender.columns[0]
    X_valid_opposite_gender = list(
        valid_data_opposite_gender[input_column_name_gender_swap]
    )
    Y_valid_opposite_gender = list(valid_data_opposite_gender[label_column_name])
    X_valid_tokenized_opposite_gender = tokenizer(
        X_valid_opposite_gender,
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )

    # Create torch dataset
    valid_dataset = Dataset(X_valid_tokenized)
    checkpoint_steps = (
        int(
            len(pd.read_csv("./data/" + args.dataset + "_train_original_gender.csv"))
            / args.batch_size_classifier
        )
        * args.num_epochs_classifier
    )
    # Define test trainer
    test_trainer = Trainer(model)
    # Make prediction
    raw_pred, _, _ = test_trainer.predict(valid_dataset)
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred[0], axis=1)

    # Load validation data for the opposite gender
    # Create torch dataset
    valid_dataset_opposite_gender = Dataset(X_valid_tokenized_opposite_gender)
    # Make prediction
    raw_pred_opposite_gender, _, _ = test_trainer.predict(valid_dataset_opposite_gender)
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
    model_path = "./saved_models/checkpoint-" + str(checkpoint_steps)
    model_before_bias_reduction = BertForSequenceClassification.from_pretrained(
        model_path, num_labels=number_of_labels, output_attentions=True
    )

    # Define test trainer
    test_trainer_before_bias_reduction = Trainer(model_before_bias_reduction)
    # Make prediction
    raw_pred, _, _ = test_trainer_before_bias_reduction.predict(valid_dataset)
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred[0], axis=1)

    # Load validation data for the opposite gender
    # Create torch dataset
    valid_dataset_opposite_gender = Dataset(X_valid_tokenized_opposite_gender)
    # Make prediction
    raw_pred_opposite_gender, _, _ = test_trainer_before_bias_reduction.predict(
        valid_dataset_opposite_gender
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

    output_file = "./output/demographic_parity_" + args.method + + "_" + args.approach + ".json"
    with open(output_file, "w+") as f:
        json.dump(str(demographic_parity), f, indent=2)

    output_file = "./output/CTF_" + args.method + + "_" + args.approach + ".json"
    with open(output_file, "w+") as f:
        json.dump(str(CTF), f, indent=2)
    # ===================================================#

    accuracy_original_gender = {}
    accuracy_opposite_gender = {}
    accuracy_overall = {}

    # Load validation data
    valid_data = pd.read_csv("./data/" + args.dataset + "_valid_original_gender.csv")
    X_valid = list(valid_data[input_column_name])
    Y_valid = list(valid_data[label_column_name])
    X_valid_tokenized = tokenizer(
        X_valid, padding=True, truncation=True, max_length=args.max_length
    )

    valid_data_opposite_gender = pd.read_csv(
        "./data/" + args.dataset + "_valid_gender_swap.csv"
    )
    X_valid_opposite_gender = list(
        valid_data_opposite_gender[input_column_name_gender_swap]
    )
    Y_valid_opposite_gender = list(valid_data_opposite_gender[label_column_name])
    X_valid_tokenized_opposite_gender = tokenizer(
        X_valid_opposite_gender,
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )

    # Create torch dataset
    valid_dataset = Dataset(X_valid_tokenized)

    # Define test trainer
    test_trainer = Trainer(model)
    # Make prediction
    raw_pred, _, _ = test_trainer.predict(valid_dataset)
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred[0], axis=1)

    # Load validation data for the opposite gender
    # Create torch dataset
    valid_dataset_opposite_gender = Dataset(X_valid_tokenized_opposite_gender)
    # Make prediction
    raw_pred_opposite_gender, _, _ = test_trainer.predict(valid_dataset_opposite_gender)
    # Preprocess raw predictions
    y_pred_opposite_gender = np.argmax(raw_pred_opposite_gender[0], axis=1)

    accuracy_original_gender["after_bias_reduction"] = torch.sum(
        torch.tensor(y_pred).double() == torch.tensor(Y_valid).double()
    ) / len(y_pred)
    accuracy_opposite_gender["after_bias_reduction"] = torch.sum(
        torch.tensor(y_pred_opposite_gender).double()
        == torch.tensor(Y_valid_opposite_gender).double()
    ) / len(y_pred_opposite_gender)

    # Make prediction
    raw_pred, _, _ = test_trainer_before_bias_reduction.predict(valid_dataset)
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred[0], axis=1)

    # Load validation data for the opposite gender
    # Create torch dataset
    valid_dataset_opposite_gender = Dataset(X_valid_tokenized_opposite_gender)
    # Make prediction
    raw_pred_opposite_gender, _, _ = test_trainer_before_bias_reduction.predict(
        valid_dataset_opposite_gender
    )
    # Preprocess raw predictions
    y_pred_opposite_gender = np.argmax(raw_pred_opposite_gender[0], axis=1)

    accuracy_original_gender["before_bias_reduction"] = torch.sum(
        torch.tensor(y_pred).double() == torch.tensor(Y_valid).double()
    ) / len(y_pred)
    accuracy_opposite_gender["before_bias_reduction"] = torch.sum(
        torch.tensor(y_pred_opposite_gender).double()
        == torch.tensor(Y_valid_opposite_gender).double()
    ) / len(y_pred_opposite_gender)

    accuracy_overall["after_bias_reduction"] = 0.5 * (
        accuracy_original_gender["after_bias_reduction"]
        + accuracy_opposite_gender["after_bias_reduction"]
    )
    accuracy_overall["before_bias_reduction"] = 0.5 * (
        accuracy_original_gender["before_bias_reduction"]
        + accuracy_opposite_gender["before_bias_reduction"]
    )

    output_file = "./output/accuracy_overall_" + args.method + + "_" + args.approach + ".json"
    with open(output_file, "w+") as f:
        json.dump(str(accuracy_overall), f, indent=2)

    output_file = "./output/accuracy_original_gender_" + args.method + + "_" + args.approach + ".json"
    with open(output_file, "w+") as f:
        json.dump(str(accuracy_original_gender), f, indent=2)

    output_file = "./output/accuracy_opposite_gender_" + args.method + + "_" + args.approach + ".json"
    with open(output_file, "w+") as f:
        json.dump(str(accuracy_opposite_gender), f, indent=2)
    # ===================================================#

    equality_of_opportunity_y_equal_0 = {}
    TNR = {}

    # Load validation data
    valid_data = pd.read_csv("./data/" + args.dataset + "_valid_original_gender.csv")
    valid_data_non_sexist_tweets = valid_data.loc[valid_data[label_column_name] == 0]
    X_valid = list(valid_data_non_sexist_tweets[input_column_name])
    X_valid_tokenized = tokenizer(
        X_valid, padding=True, truncation=True, max_length=args.max_length
    )

    valid_data_opposite_gender = pd.read_csv(
        "./data/" + args.dataset + "_valid_gender_swap.csv"
    )
    valid_data_opposite_gender_non_sexist_tweets = valid_data_opposite_gender.loc[
        valid_data_opposite_gender[label_column_name] == 0
    ]
    X_valid_opposite_gender = list(
        valid_data_opposite_gender_non_sexist_tweets[input_column_name_gender_swap]
    )
    X_valid_tokenized_opposite_gender = tokenizer(
        X_valid_opposite_gender,
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )

    # Create torch dataset
    valid_dataset = Dataset(X_valid_tokenized)
    # Define test trainer

    test_trainer = Trainer(model)
    # Make prediction
    raw_pred, _, _ = test_trainer.predict(valid_dataset)
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred[0], axis=1)

    # Load validation data for the opposite gender
    # Create torch dataset
    valid_dataset_opposite_gender = Dataset(X_valid_tokenized_opposite_gender)
    # Make prediction
    raw_pred_opposite_gender, _, _ = test_trainer.predict(valid_dataset_opposite_gender)
    # Preprocess raw predictions
    y_pred_opposite_gender = np.argmax(raw_pred_opposite_gender[0], axis=1)

    equality_of_opportunity_y_equal_0["after_bias_reduction"] = 1 - torch.abs(
        torch.mean(torch.from_numpy(y_pred).double())
        - torch.mean(torch.from_numpy(y_pred_opposite_gender).double())
    )
    TNR["after_bias_reduction"] = 1 - torch.mean(torch.from_numpy(y_pred).double())

    # Make prediction
    raw_pred, _, _ = test_trainer_before_bias_reduction.predict(valid_dataset)
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred[0], axis=1)

    # Load validation data for the opposite gender
    # Create torch dataset
    valid_dataset_opposite_gender = Dataset(X_valid_tokenized_opposite_gender)
    # Make prediction
    raw_pred_opposite_gender, _, _ = test_trainer_before_bias_reduction.predict(
        valid_dataset_opposite_gender
    )
    # Preprocess raw predictions
    y_pred_opposite_gender = np.argmax(raw_pred_opposite_gender[0], axis=1)

    equality_of_opportunity_y_equal_0["before_bias_reduction"] = 1 - torch.abs(
        torch.mean(torch.from_numpy(y_pred).double())
        - torch.mean(torch.from_numpy(y_pred_opposite_gender).double())
    )
    TNR["before_bias_reduction"] = 1 - torch.mean(torch.from_numpy(y_pred).double())

    output_file = "./output/equality_of_opportunity_y_equal_0_" + args.method + + "_" + args.approach + ".json"
    with open(output_file, "w+") as f:
        json.dump(str(equality_of_opportunity_y_equal_0), f, indent=2)

    output_file = "./output/TNR_" + args.method + + "_" + args.approach + ".json"
    with open(output_file, "w+") as f:
        json.dump(str(TNR), f, indent=2)

    # ===================================================#

    equality_of_opportunity_y_equal_1 = {}
    TPR = {}

    # Load validation data
    valid_data = pd.read_csv("./data/" + args.dataset + "_valid_original_gender.csv")
    valid_data_sexist_tweets = valid_data.loc[valid_data[label_column_name] == 1]
    X_valid = list(valid_data_sexist_tweets[input_column_name])
    X_valid_tokenized = tokenizer(
        X_valid, padding=True, truncation=True, max_length=args.max_length
    )

    valid_data_opposite_gender = pd.read_csv(
        "./data/" + args.dataset + "_valid_gender_swap.csv"
    )
    valid_data_opposite_gender_sexist_tweets = valid_data_opposite_gender.loc[
        valid_data_opposite_gender[label_column_name] == 1
    ]
    X_valid_opposite_gender = list(
        valid_data_opposite_gender_sexist_tweets[input_column_name_gender_swap]
    )
    X_valid_tokenized_opposite_gender = tokenizer(
        X_valid_opposite_gender,
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )

    # Create torch dataset
    valid_dataset = Dataset(X_valid_tokenized)

    # Define test trainer
    test_trainer = Trainer(model)
    # Make prediction
    raw_pred, _, _ = test_trainer.predict(valid_dataset)
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred[0], axis=1)

    # Load validation data for the opposite gender
    # Create torch dataset
    valid_dataset_opposite_gender = Dataset(X_valid_tokenized_opposite_gender)
    # Make prediction
    raw_pred_opposite_gender, _, _ = test_trainer.predict(valid_dataset_opposite_gender)
    # Preprocess raw predictions
    y_pred_opposite_gender = np.argmax(raw_pred_opposite_gender[0], axis=1)

    equality_of_opportunity_y_equal_1["after_bias_reduction"] = 1 - torch.abs(
        torch.mean(torch.from_numpy(y_pred).double())
        - torch.mean(torch.from_numpy(y_pred_opposite_gender).double())
    )
    TPR["after_bias_reduction"] = torch.mean(torch.from_numpy(y_pred).double())

    # Make prediction
    raw_pred, _, _ = test_trainer_before_bias_reduction.predict(valid_dataset)
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred[0], axis=1)

    # Load validation data for the opposite gender
    # Create torch dataset
    valid_dataset_opposite_gender = Dataset(X_valid_tokenized_opposite_gender)
    # Make prediction
    raw_pred_opposite_gender, _, _ = test_trainer_before_bias_reduction.predict(
        valid_dataset_opposite_gender
    )
    # Preprocess raw predictions
    y_pred_opposite_gender = np.argmax(raw_pred_opposite_gender[0], axis=1)

    equality_of_opportunity_y_equal_1["before_bias_reduction"] = 1 - torch.abs(
        torch.mean(torch.from_numpy(y_pred).double())
        - torch.mean(torch.from_numpy(y_pred_opposite_gender).double())
    )
    TPR["before_bias_reduction"] = torch.mean(torch.from_numpy(y_pred).double())

    output_file = "./output/equality_of_opportunity_y_equal_1_" + args.method + + "_" + args.approach + ".json"
    with open(output_file, "w+") as f:
        json.dump(str(equality_of_opportunity_y_equal_1), f, indent=2)

    output_file = "./output/TPR_" + args.method + + "_" + args.approach + ".json"
    with open(output_file, "w+") as f:
        json.dump(str(TPR), f, indent=2)
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
    output_file = "./output/equality_of_odds_" + args.method + + "_" + args.approach + ".json"
    with open(output_file, "w+") as f:
        json.dump(str(equality_of_odds), f, indent=2)
    # ===================================================#


# Some of the following parts are taken from https://towardsdatascience.com/fine-tuning-pretrained-nlp-models-with-huggingfaces-trainer-6326a4456e7b by Vincent Tan
def train_classifier(args,data_augmentation_flag=None):
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
        model: the model that is going to be our starting point for policy
        gradient
        tokenizer: the tokenizer used before giving the sentences to the
        classifier model
    """
    # Load the dataset
    train_dataset, val_dataset, test_dataset = data_loader(args,apply_data_augmentation=data_augmentation_flag)
    # The number of epochs afterwhich we save the model. We set it to this value to only save the last model.
    checkpoint_steps = (
        int(train_dataset.__len__()/ args.batch_size_classifier) * args.num_epochs_classifier
    )

    if args.load_pretrained_classifier:

        tokenizer = BertTokenizer.from_pretrained(args.classifier_model)
        model = BertForSequenceClassification.from_pretrained(
            args.model_path + str(checkpoint_steps),
            num_labels=len(set(train_dataset.labels)),
            output_attentions=True,
        )

    else:
        # Define pretrained tokenizer and model
        model_name = args.classifier_model
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(set(train_dataset.labels)),
            output_attentions=True,
        )


        # Define Trainer parameters

        # Define Trainer
        classifier_args = TrainingArguments(
            output_dir=args.output_dir,
            evaluation_strategy="steps",
            eval_steps=checkpoint_steps,
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
            compute_metrics=measure_performance_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Train pre-trained model
        trainer.train()
    return model, tokenizer


def measure_performance_metrics(p):
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
