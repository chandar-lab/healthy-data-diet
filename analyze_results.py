import pandas as pd
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
import torch
from classifier import Dataset, measure_performance_metrics
from argparse import ArgumentParser


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softmax = torch.nn.Softmax(dim=1).to(device)


def compute_confidence_and_variability_after_debiasing(args, data):
    """
    Compute the confidence and variability in the model after debiasing as in https://arxiv.org/pdf/2009.10795.pdf as follows:
    1) The model is trained for multple epochs, and after each epoch it is used to give predictions for a specific dataset (it could be training or validation).
    2) The examples in the dataset are categorized into "easy-to-learn", "hard-to-learn" and "ambiguous" based on the mean and standard deviation in the predictions of the ground truth labels.
    3) "easy-to-learn" examples are those that the model predicts correctly over multiple epochs, while low variability. "hard-to-learn" examples are those that the model incorrectly predicts with low variability, and "ambiguous" examples are those with high variability in the prediction.
    args:
        args: the arguments given by the user
        data: the csv file of the dataset for which we compute the confidence and variability.
    returns:
        the function returns:
        confidence: the mean of the predictions that the debiased model gives to the ground truth label, over multiple epochs.
        variability: the standard deviation of the predictions that the debiased model gives to the groud truth label, over multiple epochs.
    """
    tokenizer = BertTokenizer.from_pretrained(args.classifier_model)
    data_train = pd.read_csv("./data/" + args.dataset + "_train_original_gender.csv")
    data_valid = pd.read_csv("./data/" + args.dataset + "_valid_original_gender.csv")

    # ----- 1. Preprocess data -----#
    # Preprocess data
    X_val = list(data_valid[data_valid.columns[1]])
    y_val = list(data_valid[data_valid.columns[2]])

    X_val_tokenized = tokenizer(
        X_val, padding=True, truncation=True, max_length=args.max_length
    )

    valid_dataset = Dataset(X_val_tokenized, y_val)

    tokenizer = BertTokenizer.from_pretrained(args.classifier_model)
    prediction = []

    model_after_debiasing = BertForSequenceClassification.from_pretrained(
        args.classifier_model,
        num_labels=len(data_train.Class.unique()),
        output_attentions=True,
    )

    for i in range(args.num_saved_debiased_models):

        model_after_debiasing.load_state_dict(
            torch.load(
                "./saved_models/"
                + args.classifier_model
                + "_debiased_epoch_"
                + str(i)
                + ".pt",
                map_location=device,
            )
        )
        # Define test trainer
        test_trainer_after_debiasing = Trainer(model_after_debiasing)

        # Save the predictions after each epoch (based on the paper https://arxiv.org/pdf/2009.10795.pdf)
        prediction.append(
            softmax(
                torch.tensor(test_trainer_after_debiasing.predict(valid_dataset)[0][0])
            )
        )
    ground_truth_labels = torch.tensor(y_val).to(device)

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

    return (
        confidence,
        variability,
    )


def compute_confidence_and_variability_before_debiasing(args, data):
    """
    Compute the confidence and variability in the model before debiasing as in https://arxiv.org/pdf/2009.10795.pdf as follows:
    1) The model is trained for multple epochs, and after each epoch it is used to give predictions for a specific dataset (it could be training or validation).
    2) The examples in the dataset are categorized into "easy-to-learn", "hard-to-learn" and "ambiguous" based on the mean and standard deviation in the predictions of the ground truth labels.
    3) "easy-to-learn" examples are those that the model predicts correctly over multiple epochs, while low variability. "hard-to-learn" examples are those that the model incorrectly predicts with low variability, and "ambiguous" examples are those with high variability in the prediction.
    args:
        args: the arguments given by the user
        data: the csv file of the dataset for which we compute the confidence and variability.
    returns:
        the function returns:
        confidence: the mean of the predictions that the model gives to the ground truth label, over multiple epochs.
        variability: the standard deviation of the predictions that the model gives to the groud truth label, over multiple epochs.
        test_trainer_before_debiasing:  used get the predictions of the model before_debiasing.
        valid_dataset: the dataset for which we compute the confidence and variability.
        tokenizer: the tokenizer used by the biased model
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
    valid_dataset = Dataset(X_val_tokenized, y_val)
    # Save the model weights after each epoch
    checkpoint_steps = int(len(data_train) / args.batch_size_classifier)

    # Define Trainer parameters
    classifier_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=checkpoint_steps,
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
            model_path = "./saved_models/checkpoint-" + str(checkpoint_steps * i)
            model_before_debiasing = BertForSequenceClassification.from_pretrained(
                model_path,
                num_labels=len(data_train.Class.unique()),
                output_attentions=True,
            )
        else:
            model_name = args.classifier_model
            model_before_debiasing = BertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(data_train.Class.unique()),
                output_attentions=True,
            )
        # Define Trainer
        trainer = Trainer(
            model=model_before_debiasing,
            args=classifier_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=measure_performance_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Train pre-trained model for 1 epoch
        trainer.train()
        # Define test trainer
        test_trainer_before_debiasing = Trainer(model_before_debiasing)

        # Save the predictions after each epoch (based on the paper https://arxiv.org/pdf/2009.10795.pdf)
        prediction.append(
            softmax(
                torch.tensor(test_trainer_before_debiasing.predict(valid_dataset)[0][0])
            )
        )

    ground_truth_labels = torch.tensor(y_val).to(device)

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

    return (
        confidence,
        variability,
        test_trainer_before_debiasing,
        valid_dataset,
        tokenizer,
    )


def log_topk_attention_tokens(
    args,
    data,
    test_trainer_before_debiasing,
    test_trainer_after_debiasing,
    valid_dataset,
    tokenizer,
):
    """
    Log the top k tokens to which the classification token (CLS) attends
    args:
        args: the arguments given by the user
        data: the csv file of the dataset for which we compute the confidence and variability.
        test_trainer_before_debiasing: the test trainer that is used to get the predictions of the model before debiasing
        test_trainer_after_debiasing: the test trainer that is used to get the predictions of the model after debiasing
        valid_dataset: the dataset for which we compute the top k attention tokens.
    returns:
        the function doesnt return anything, since the top k tokens are added to the csv file.
    """
    # Compute the attention weights in the last layer of the biased model
    top_attention_tokens_biased = []
    valid_ids = valid_dataset[:]["input_ids"].to(device)
    last_layer_attention_before_debiasing = torch.tensor(
        (test_trainer_before_debiasing.predict(valid_dataset)[0][-1][-1])
    ).to(device)

    # ===================================================#

    # Log the top k tokens that the classification token attends to in the last layer of the biased and de-biased models for all the heads combined
    number_of_heads = last_layer_attention_before_debiasing.shape[1]
    top_attention_tokens_debiased = []
    last_layer_attention_after_debiasing = torch.tensor(
        (test_trainer_after_debiasing.predict(valid_dataset)[0][-1][-1])
    ).to(device)

    top_attention_tokens_biased.append(
        [
            [
                tokenizer.convert_ids_to_tokens(valid_ids[j])[i]
                for i in torch.topk(
                    torch.sum(
                        last_layer_attention_before_debiasing[j][0:number_of_heads],
                        dim=0,
                    )[0],
                    args.num_tokens_logged,
                )[1]
            ]
            for j in range(len(valid_dataset))
        ]
    )
    top_attention_tokens_debiased.append(
        [
            [
                tokenizer.convert_ids_to_tokens(valid_ids[j])[i]
                for i in torch.topk(
                    torch.sum(
                        last_layer_attention_after_debiasing[j][0:number_of_heads],
                        dim=0,
                    )[0],
                    args.num_tokens_logged,
                )[1]
            ]
            for j in range(len(valid_dataset))
        ]
    )
    data["top_attention_tokens_biased_" + "all_heads"] = top_attention_tokens_biased[0]
    data[
        "top_attention_tokens_de-biased_" + "all_heads"
    ] = top_attention_tokens_debiased[0]
    top_attention_tokens_debiased = []
    top_attention_tokens_biased = []

    # ===================================================#

    # Log the top k tokens that the classification token attends to in the last layer of the biased and de-biased models for each attention head
    if args.log_top_tokens_each_head == True:
        for model_head in range(number_of_heads):
            top_attention_tokens_biased.append(
                [
                    [
                        tokenizer.convert_ids_to_tokens(valid_ids[j])[i]
                        for i in torch.topk(
                            last_layer_attention_before_debiasing[j][model_head][0],
                            args.num_tokens_logged,
                        )[1]
                    ]
                    for j in range(len(valid_dataset))
                ]
            )
            top_attention_tokens_debiased.append(
                [
                    [
                        tokenizer.convert_ids_to_tokens(valid_ids[j])[i]
                        for i in torch.topk(
                            last_layer_attention_after_debiasing[j][model_head][0],
                            args.num_tokens_logged,
                        )[1]
                    ]
                    for j in range(len(valid_dataset))
                ]
            )
            data[
                "top_attention_tokens_biased_" + "head_" + str(model_head)
            ] = top_attention_tokens_biased[0]
            data[
                "top_attention_tokens_de-biased_" + "head_" + str(model_head)
            ] = top_attention_tokens_debiased[0]

    return data


def analyze_results(args):
    """
    Analyze the results on the validation data by focusing on:
    1) Attention weights: We log the top k tokens to which the classification token (CLS) attends before and after de-biasing.
    2) Type of examples: We follow the procedure in https://arxiv.org/pdf/2009.10795.pdf where the examples
        are categorized into "easy-to-learn", "hard-to-learn" and "ambiguous". The intuition is to know which category is mostly affected by the de-biasing algorithm.
    args:
        args: the arguments given by the user
    returns:
        the function doesnt return anything, since the output is written in a csv file.
    """
    model_name = args.classifier_model
    # Load the data that we need to analyze
    data = pd.read_csv("./data/" + args.dataset + "_valid_original_gender.csv")
    label_column_name = data.columns[2]
    ground_truth_labels = torch.tensor(list(data[label_column_name])).to(device)
    # Compute the confidence and variability as in https://arxiv.org/pdf/2009.10795.pdf
    (
        confidence_before_debiasing,
        variability_before_debiasing,
        test_trainer_before_debiasing,
        valid_dataset,
        tokenizer,
    ) = compute_confidence_and_variability_before_debiasing(args, data)

    (
        confidence_after_debiasing,
        variability_after_debiasing,
    ) = compute_confidence_and_variability_after_debiasing(args, data)

    # Get te output of the model before debiasing
    y_pred_before_debiasing = torch.argmax(
        softmax(
            torch.tensor(test_trainer_before_debiasing.predict(valid_dataset)[0][0])
        ),
        axis=1,
    )

    # Load the model after debiasing
    model_after_debiasing = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=len(data.Class.unique()), output_attentions=True
    )
    model_after_debiasing.load_state_dict(
        torch.load(
            "./saved_models/" + args.classifier_model + "_debiased_best.pt",
            map_location=device,
        )
    )
    # Define test trainer for the biased model
    test_trainer_after_debiasing = Trainer(model_after_debiasing)

    # Load the de-biased model to compare its performance to the biased one
    prediction_after_debiasing = []
    prediction_after_debiasing = softmax(
        torch.tensor(test_trainer_after_debiasing.predict(valid_dataset)[0][0])
    )
    # Get te output of the model after debiasing
    y_pred_after_debiasing = torch.argmax(prediction_after_debiasing, axis=1)

    data = log_topk_attention_tokens(
        args,
        data,
        test_trainer_before_debiasing,
        test_trainer_after_debiasing,
        valid_dataset,
        tokenizer,
    )
    # ===================================================#

    # To analyze our results, we keep track of the confidence and variability in prediction of each example in the validation data, as well as whether or not
    # it is correctly classified before and after de-biasing.
    data["confidence_before_debiasing"] = list(
        confidence_before_debiasing.cpu().detach().numpy()
    )
    data["variability_before_debiasing"] = list(
        variability_before_debiasing.cpu().detach().numpy()
    )
    data["confidence_after_debiasing"] = list(
        confidence_after_debiasing.cpu().detach().numpy()
    )
    data["variability_after_debiasing"] = list(
        variability_after_debiasing.cpu().detach().numpy()
    )
    data["Correct classification? before debiasing"] = (
        ground_truth_labels.cpu() == y_pred_before_debiasing
    )
    data["Correct classification? after debiasing"] = (
        ground_truth_labels.cpu() == y_pred_after_debiasing
    )
    data.to_csv("./output/data_analysis.csv", index=False)


def parse_args():
    """Parses the command line arguments."""
    parser = ArgumentParser()
    # arguments for analysing the data
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
    parser.add_argument(
        "--log_top_tokens_each_head",
        type=bool,
        default=False,
        help="Whether or not to log the top k tokens that the classification token attends to in the last layer for each attention head",
    )
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
    parser.add_argument(
        "--output_dir",
        default="saved_models",
        help="Directory to saved models",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze_results(args)
