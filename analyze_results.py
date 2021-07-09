import pandas as pd
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
import torch
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from classifier import Dataset, compute_metrics
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softmax = torch.nn.Softmax(dim=1).to(device)
parser = argparse.ArgumentParser()


def analyze_results(args):
    """
    Analyze the results by focusing on:
    1) Attention weights: We log the top 5 tokens to which the classification token (CLS) attends before and after de-biasing.
    2) Type of examples: We follow the procedure in https://arxiv.org/pdf/2009.10795.pdf where the examples
        are categorized into "easy-to-learn", "hard-to-learn" and "ambiguous". The intuition is to know which category is mostly affected by the de-biasing algorithm.
    args:
        args: the arguments given by the user
    returns:
        the function doesnt return anything, since the output is written in a csv file.
    """
    tokenizer = BertTokenizer.from_pretrained(args.classifier_model)
    data_train = pd.read_csv("./data/" + args.dataset + "_train_original_gender.csv")
    data_valid = pd.read_csv("./data/" + args.dataset + "_valid_original_gender.csv")

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
    val_steps = int(len(X_val) / args.batch_size_classifier)

    # Define Trainer parameters
    classifier_args = TrainingArguments(
        output_dir="saved_models",
        evaluation_strategy="steps",
        eval_steps=val_steps,
        save_steps=3000,
        per_device_train_batch_size=args.batch_size_classifier,
        per_device_eval_batch_size=args.batch_size_classifier,
        num_train_epochs=1,
        load_best_model_at_end=True,
    )

    tokenizer = BertTokenizer.from_pretrained(args.classifier_model)
    prediction = []

    for i in range(args.num_epochs_classifier):
        if i != 0:
            # If this is not the first epoch, we load the model we saved from the previous epoch
            model_path = "./saved_models/checkpoint-" + str(i * val_steps)
            model = BertForSequenceClassification.from_pretrained(
                model_path,
                num_labels=len(data_train.Class.unique()),
                output_attentions=True,
            )
        else:
            model_name = args.classifier_model
            model = BertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(data_train.Class.unique()),
                output_attentions=True,
            )
        # Define Trainer    
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
        # Define test trainer
        test_trainer = Trainer(model)

        # Load test data
        test_data = pd.read_csv("./data/" + args.dataset + "_valid_original_gender.csv")
        input_column_name = test_data.columns[1]
        label_column_name = test_data.columns[2]
        X_test = list(test_data[input_column_name])
        X_test_tokenized = tokenizer(
            X_test, padding=True, truncation=True, max_length=args.max_length
        )

        # Create torch dataset
        test_dataset = Dataset(X_test_tokenized)
        # Make prediction
        prediction.append(
            softmax(torch.tensor(test_trainer.predict(test_dataset)[0][0]))
        )

    y_pred = torch.argmax(prediction[-1], axis=1)
    ground_truth_labels = torch.tensor(list(test_data[label_column_name])).to(device)

    prediction_all = torch.cat(
        [torch.unsqueeze(prediction[i], dim=0) for i in range(len(prediction))]
    )
    prediction_mean = torch.mean(prediction_all, dim=0).to(device)
    prediction_deviation = torch.std(prediction_all, dim=0).to(device)

    confidence = torch.tensor(
        [
            prediction_mean[i, int(ground_truth_labels[i])]
            for i in range(ground_truth_labels.shape[0])
        ]
    ).to(device)
    variability = torch.tensor(
        [
            prediction_deviation[i, int(ground_truth_labels[i])]
            for i in range(ground_truth_labels.shape[0])
        ]
    ).to(device)

    # Compute the attention weights in the last layer of the biased model
    top5_attention_tokens = []
    test_ids = test_dataset[:]["input_ids"].to(device)
    last_layer_attention = torch.tensor(
        (test_trainer.predict(test_dataset)[0][-1][-1])
    ).to(device)

    # ===================================================#

    # Load the de-biased model to compare its performance to the biased one
    prediction_after_debiasing = []
    model_after_debiasing = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=len(data_train.Class.unique()), output_attentions=True
    )
    model_after_debiasing.load_state_dict(
        torch.load(
            "./saved_models/" + args.classifier_model + "_debiased.pt",
            map_location=device,
        )
    )
    # Define test trainer
    test_trainer = Trainer(model_after_debiasing)
    prediction_after_debiasing = softmax(
        torch.tensor(test_trainer.predict(test_dataset)[0][0])
    )
    y_pred_after_debiasing = torch.argmax(prediction_after_debiasing, axis=1)

    # ===================================================#

    # Log the top 5 tokens that the classification token attends to in the last layer of the biased and de-biased models
    top5_attention_tokens_debiased = []
    last_layer_attention_after_debiasing = torch.tensor(
        (test_trainer.predict(test_dataset)[0][-1][-1])
    ).to(device)
    for model_head in range(last_layer_attention.shape[1]):
        top5_attention_tokens.append(
            [
                [
                    tokenizer.convert_ids_to_tokens(test_ids[j])[i]
                    for i in torch.topk(last_layer_attention[j][model_head][0], 5)[1]
                ]
                for j in range(len(test_dataset))
            ]
        )
        top5_attention_tokens_debiased.append(
            [
                [
                    tokenizer.convert_ids_to_tokens(test_ids[j])[i]
                    for i in torch.topk(
                        last_layer_attention_after_debiasing[j][model_head][0], 5
                    )[1]
                ]
                for j in range(len(test_dataset))
            ]
        )
        test_data[
            "top5_attention_tokens_biased_" + "head_" + str(model_head)
        ] = top5_attention_tokens[0]
        test_data[
            "top5_attention_tokens_de-biased_" + "head_" + str(model_head)
        ] = top5_attention_tokens_debiased[0]
        top5_attention_tokens_debiased = []
        top5_attention_tokens = []

    # ===================================================#

    # To analyze our results, we keep track of the confidence and variability in prediction of each example in the test data, as well as whether or not
    # it is correctly classified before and after de-biasing.
    test_data["confidence"] = list(confidence.cpu().detach().numpy())
    test_data["variability"] = list(variability.cpu().detach().numpy())
    test_data["Correct classification? before debiasing"] = (
        ground_truth_labels.cpu() == y_pred
    )
    test_data["Correct classification? after debiasing"] = (
        ground_truth_labels.cpu() == y_pred_after_debiasing
    )
    test_data.to_csv("./output/data_analysis.csv", index=False)


# arguments for the classifier
parser.add_argument(
    "--classifier_model",
    choices=["bert-base-uncased"],
    default="bert-base-uncased",
    help="Type of classifier used",
)
parser.add_argument(
    "--dataset",
    choices=["Equity-Evaluation-Corpus,twitter_dataset"],
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
    "--max_length",
    type=int,
    default=512,
    help="The maximum length of the sentences that we classify (in terms of the number of tokens)",
)


if __name__ == "__main__":
    args = parser.parse_args()
    analyze_results(args)
