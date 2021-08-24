from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from models.data_loader import data_loader
import torch
import json
import numpy as np
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def measure_bias_metrics(model_after_bias_reduction, dataset, args):
    """
    Compute metrics that measure the bias before and after applying our
    de-biasing algorithm.
    The metrics that we compute are the following:
    1) Demagraphic parity
    2) Equality of odds
    3) Counterfactual token fairness
    4) True negative rate
    5) True positive rate
    6) Accuracy for both the original and opposite gender (each separately
    and combined)
    7) Equality of opportunity
    args:
        args: the arguments given by the user
        model: the model after updating its weights based on the policy gradient
        algorithm.
        tokenizer: the tokenizer used before giving the sentences to the classifier
    returns:
        the function doesnt return anything, since all the metrics are saved
        in json files.
    """
    demographic_parity = {}
    
    y_pred_after_bias_reduction = {}
    y_pred_before_bias_reduction = {}
    
    y_pred_opposite_gender_after_bias_reduction = {}
    y_pred_opposite_gender_before_bias_reduction = {}
    
    checkpoint_steps = (
        int(
            len(pd.read_csv("./data/" + args.dataset + "_train_original_gender.csv"))
            / args.batch_size_pretraining
        )
        * args.num_epochs_pretraining
    )

    # Load trained model before bias reduction
    model_path = "./saved_models/checkpoint-" + str(checkpoint_steps)
    model_before_bias_reduction = BertForSequenceClassification.from_pretrained(
        model_path, num_labels=len(set(dataset.labels)), output_attentions=True
    )    
    

    for i in range(int(np.ceil(dataset.__len__() / args.batch_size))):

        results_original_gender_after_bias_reduction = model_after_bias_reduction.forward(
            input_ids=torch.tensor(
                dataset.encodings["input_ids"][
                    i * args.batch_size : (i + 1) * args.batch_size
                ]
            ).to(device),
            attention_mask=torch.tensor(
                dataset.encodings["attention_mask"][
                    i * args.batch_size : (i + 1) * args.batch_size
                ]
            ).to(device),
            token_type_ids=torch.tensor(
                dataset.encodings["token_type_ids"][
                    i * args.batch_size : (i + 1) * args.batch_size
                ]
            ).to(device),
        )[0]
        results_gender_swap_after_bias_reduction = model_after_bias_reduction.forward(
            input_ids=torch.tensor(
                dataset.encodings_gender_swap["input_ids"][
                    i * args.batch_size : (i + 1) * args.batch_size
                ]
            ).to(device),
            attention_mask=torch.tensor(
                dataset.encodings_gender_swap["attention_mask"][
                    i * args.batch_size : (i + 1) * args.batch_size
                ]
            ).to(device),
            token_type_ids=torch.tensor(
                dataset.encodings_gender_swap["token_type_ids"][
                    i * args.batch_size : (i + 1) * args.batch_size
                ]
            ).to(device),
        )[0]
        
        results_original_gender_before_bias_reduction = model_before_bias_reduction.forward(
            input_ids=torch.tensor(
                dataset.encodings["input_ids"][
                    i * args.batch_size : (i + 1) * args.batch_size
                ]
            ).to(device),
            attention_mask=torch.tensor(
                dataset.encodings["attention_mask"][
                    i * args.batch_size : (i + 1) * args.batch_size
                ]
            ).to(device),
            token_type_ids=torch.tensor(
                dataset.encodings["token_type_ids"][
                    i * args.batch_size : (i + 1) * args.batch_size
                ]
            ).to(device),
        )[0]
        results_gender_swap_before_bias_reduction = model_before_bias_reduction.forward(
            input_ids=torch.tensor(
                dataset.encodings_gender_swap["input_ids"][
                    i * args.batch_size : (i + 1) * args.batch_size
                ]
            ).to(device),
            attention_mask=torch.tensor(
                dataset.encodings_gender_swap["attention_mask"][
                    i * args.batch_size : (i + 1) * args.batch_size
                ]
            ).to(device),
            token_type_ids=torch.tensor(
                dataset.encodings_gender_swap["token_type_ids"][
                    i * args.batch_size : (i + 1) * args.batch_size
                ]
            ).to(device),
        )[0]        
        
        y_pred_after_bias_reduction.append(results_original_gender_after_bias_reduction)
        y_pred_opposite_gender_after_bias_reduction.append(results_gender_swap_after_bias_reduction)

        y_pred_before_bias_reduction.append(results_original_gender_before_bias_reduction)
        y_pred_opposite_gender_before_bias_reduction.append(results_gender_swap_before_bias_reduction)        
        
    demographic_parity["after_bias_reduction"] = 1 - torch.abs(
        torch.mean(y_pred_after_bias_reduction.double())
        - torch.mean(y_pred_opposite_gender_after_bias_reduction.double())
    )    
    
        
    demographic_parity["before_bias_reduction"] = 1 - torch.abs(
        torch.mean(y_pred_before_bias_reduction.double())
        - torch.mean(y_pred_opposite_gender_before_bias_reduction.double())
    )        
    
    output_file = "./output/demographic_parity_" + args.method + + "_" + args.approach + + "_" + args.dataset + ".json"
    with open(output_file, "w+") as f:
        json.dump(str(demographic_parity), f, indent=2)    
    


# Some of the following parts are taken from
# https://towardsdatascience.com/fine-tuning-pretrained-nlp-models-with-huggingfaces-trainer-6326a4456e7b
# by Vincent Tan
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
        model: the model that is going to be our starting point for polic
        y gradient
        tokenizer: the tokenizer used before giving the sentences to the
        classifier model
    """
    # Load the dataset
    train_dataset, val_dataset, test_dataset = data_loader(args,apply_data_augmentation=data_augmentation_flag)
    # The number of epochs afterwhich we save the model. We set it to this
    #value to only save the last model.
    checkpoint_steps = (
        int(train_dataset.__len__()/ args.batch_size_pretraining) * args.num_epochs_pretraining
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
