import pandas as pd
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score



def train_classifier(args):

    # Define pretrained tokenizer and model
    model_name = args.classifier_model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4)

    # Read data
    data = pd.read_csv('./data/'+args.dataset+'_train_original_gender.csv')


    # ----- 1. Preprocess data -----#
    # Preprocess data
    X = list(data["Tweets"])
    y = list(data["Class"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=100)
    X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=100)

    train_dataset = Dataset(X_train_tokenized, y_train)
    val_dataset = Dataset(X_val_tokenized, y_val)

    # Define Trainer parameters

    # Define Trainer
    classifier_args = TrainingArguments(
        output_dir="output",
        evaluation_strategy="steps",
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
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred,average='micro')
    precision = precision_score(y_true=labels, y_pred=pred,average='micro')
    f1 = f1_score(y_true=labels, y_pred=pred,average='micro')

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