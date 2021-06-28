import pandas as pd
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
from sklearn.model_selection import train_test_split
import torch
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softmax=torch.nn.Softmax(dim=1).to(device)

def measure_bias_metrics(model,tokenizer,args):


    demographic_parity = {}
    CTF = {}
    # Load test data
    test_data = pd.read_csv('./data/'+args.dataset+'_valid_original_gender.csv')
    X_test = list(test_data["Tweets"])
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)
    
    test_data_opposite_gender = pd.read_csv('./data/'+args.dataset+'_valid_reversed_gender.csv')
    X_test_opposite_gender = list(test_data_opposite_gender["Tweets2"])
    X_test_tokenized_opposite_gender = tokenizer(X_test_opposite_gender, padding=True, truncation=True, max_length=512)
    
    # Create torch dataset
    test_dataset = Dataset(X_test_tokenized)
    # Define test trainer
    test_trainer = Trainer(model)
    # Make prediction
    raw_pred, _, _ = test_trainer.predict(test_dataset)
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)
    
    
    # Load test data for the opposite gender
    
    # Create torch dataset
    test_dataset_opposite_gender = Dataset(X_test_tokenized_opposite_gender)
    # Make prediction
    raw_pred_opposite_gender, _, _ = test_trainer.predict(test_dataset_opposite_gender)
    # Preprocess raw predictions
    y_pred_opposite_gender = np.argmax(raw_pred_opposite_gender, axis=1)
    
    demographic_parity['after_bias_reduction'] = 1-torch.abs(torch.mean(torch.from_numpy(y_pred).double()) - torch.mean(torch.from_numpy(y_pred_opposite_gender).double()))
    CTF['after_bias_reduction'] = torch.sum(torch.abs(softmax(torch.from_numpy(raw_pred))[:,1] - softmax(torch.from_numpy(raw_pred_opposite_gender))[:,1]))

    # We also compute the same metric before reducing the bias
    # Load trained model
    model_path =  "./output/checkpoint-500"
    model_before_bias_reduction = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
    
    # Define test trainer
    test_trainer_before_bias_reduction = Trainer(model_before_bias_reduction)
    # Make prediction
    raw_pred, _, _ = test_trainer_before_bias_reduction.predict(test_dataset)
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)

    # Load test data for the opposite gender
    # Create torch dataset
    test_dataset_opposite_gender = Dataset(X_test_tokenized_opposite_gender)    
    # Make prediction
    raw_pred_opposite_gender, _, _ = test_trainer_before_bias_reduction.predict(test_dataset_opposite_gender)
    # Preprocess raw predictions
    y_pred_opposite_gender = np.argmax(raw_pred_opposite_gender, axis=1)
    
    demographic_parity['before_bias_reduction'] = 1-torch.abs(torch.mean(torch.from_numpy(y_pred).double()) - torch.mean(torch.from_numpy(y_pred_opposite_gender).double()))
    CTF['before_bias_reduction'] = torch.sum(torch.abs(softmax(torch.from_numpy(raw_pred))[:,1] - softmax(torch.from_numpy(raw_pred_opposite_gender))[:,1]))


    output_file = "./output/demographic_parity.txt"
    f = open(output_file, "w")
    f.write(str(demographic_parity))
    f.close()

    output_file = "./output/CTF.txt"
    f = open(output_file, "w")
    f.write(str(CTF))
    f.close()
#===================================================#

    equality_of_opportunity_y_equal_0 = {}
    TNR = {}
    # Load test data
    test_data = pd.read_csv('./data/'+args.dataset+'_valid_original_gender.csv')
    test_data_non_sexist_tweets = test_data.loc[test_data['Class'] == 0]
    X_test = list(test_data_non_sexist_tweets["Tweets"])
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)
    
    test_data_opposite_gender = pd.read_csv('./data/'+args.dataset+'_valid_reversed_gender.csv')
    test_data_opposite_gender_non_sexist_tweets = test_data_opposite_gender.loc[test_data_opposite_gender['Class'] == 0]
    X_test_opposite_gender = list(test_data_opposite_gender_non_sexist_tweets["Tweets2"])
    X_test_tokenized_opposite_gender = tokenizer(X_test_opposite_gender, padding=True, truncation=True, max_length=512)
    
    # Create torch dataset
    test_dataset = Dataset(X_test_tokenized)
    # Define test trainer
    test_trainer = Trainer(model)
    # Make prediction
    raw_pred, _, _ = test_trainer.predict(test_dataset)
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)
    
    
    # Load test data for the opposite gender
    
    # Create torch dataset
    test_dataset_opposite_gender = Dataset(X_test_tokenized_opposite_gender)
    # Make prediction
    raw_pred_opposite_gender, _, _ = test_trainer.predict(test_dataset_opposite_gender)
    # Preprocess raw predictions
    y_pred_opposite_gender = np.argmax(raw_pred_opposite_gender, axis=1)
    
    equality_of_opportunity_y_equal_0['after_bias_reduction'] = 1-torch.abs(torch.mean(torch.from_numpy(y_pred).double()) - torch.mean(torch.from_numpy(y_pred_opposite_gender).double()))
    TNR['after_bias_reduction'] = 1-torch.mean(torch.from_numpy(y_pred).double())

    # Make prediction
    raw_pred, _, _ = test_trainer_before_bias_reduction.predict(test_dataset)
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)

    # Load test data for the opposite gender
    # Create torch dataset
    test_dataset_opposite_gender = Dataset(X_test_tokenized_opposite_gender)    
    # Make prediction
    raw_pred_opposite_gender, _, _ = test_trainer_before_bias_reduction.predict(test_dataset_opposite_gender)
    # Preprocess raw predictions
    y_pred_opposite_gender = np.argmax(raw_pred_opposite_gender, axis=1)
    
    equality_of_opportunity_y_equal_0['before_bias_reduction'] = 1-torch.abs(torch.mean(torch.from_numpy(y_pred).double()) - torch.mean(torch.from_numpy(y_pred_opposite_gender).double()))
    TNR['before_bias_reduction'] = 1-torch.mean(torch.from_numpy(y_pred).double())

    output_file = "./output/equality_of_opportunity_y_equal_0.txt"
    f = open(output_file, "w")
    f.write(str(equality_of_opportunity_y_equal_0))
    f.close()

    output_file = "./output/TNR.txt"
    f = open(output_file, "w")
    f.write(str(TNR))
    f.close()    

    #===================================================#

    equality_of_opportunity_y_equal_1 = {}
    TPR = {}
    # Load test data
    test_data = pd.read_csv('./data/'+args.dataset+'_valid_original_gender.csv')
    test_data_sexist_tweets = test_data.loc[test_data['Class'] == 1]
    X_test = list(test_data_sexist_tweets["Tweets"])
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)
    
    test_data_opposite_gender = pd.read_csv('./data/'+args.dataset+'_valid_reversed_gender.csv')
    test_data_opposite_gender_sexist_tweets = test_data_opposite_gender.loc[test_data_opposite_gender['Class'] == 1]
    X_test_opposite_gender = list(test_data_opposite_gender_sexist_tweets["Tweets2"])
    X_test_tokenized_opposite_gender = tokenizer(X_test_opposite_gender, padding=True, truncation=True, max_length=512)
    
    # Create torch dataset
    test_dataset = Dataset(X_test_tokenized)
    # Define test trainer
    test_trainer = Trainer(model)
    # Make prediction
    raw_pred, _, _ = test_trainer.predict(test_dataset)
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)
    
    
    # Load test data for the opposite gender
    
    # Create torch dataset
    test_dataset_opposite_gender = Dataset(X_test_tokenized_opposite_gender)
    # Make prediction
    raw_pred_opposite_gender, _, _ = test_trainer.predict(test_dataset_opposite_gender)
    # Preprocess raw predictions
    y_pred_opposite_gender = np.argmax(raw_pred_opposite_gender, axis=1)
    
    equality_of_opportunity_y_equal_1['after_bias_reduction'] = 1-torch.abs(torch.mean(torch.from_numpy(y_pred).double()) - torch.mean(torch.from_numpy(y_pred_opposite_gender).double()))
    TPR['after_bias_reduction'] = torch.mean(torch.from_numpy(y_pred).double())

    # Make prediction
    raw_pred, _, _ = test_trainer_before_bias_reduction.predict(test_dataset)
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)

    # Load test data for the opposite gender
    # Create torch dataset
    test_dataset_opposite_gender = Dataset(X_test_tokenized_opposite_gender)    
    # Make prediction
    raw_pred_opposite_gender, _, _ = test_trainer_before_bias_reduction.predict(test_dataset_opposite_gender)
    # Preprocess raw predictions
    y_pred_opposite_gender = np.argmax(raw_pred_opposite_gender, axis=1)
    
    equality_of_opportunity_y_equal_1['before_bias_reduction'] = 1-torch.abs(torch.mean(torch.from_numpy(y_pred).double()) - torch.mean(torch.from_numpy(y_pred_opposite_gender).double()))
    TPR['before_bias_reduction'] = torch.mean(torch.from_numpy(y_pred).double())

    output_file = "./output/equality_of_opportunity_y_equal_1.txt"
    f = open(output_file, "w")
    f.write(str(equality_of_opportunity_y_equal_1))
    f.close()

    output_file = "./output/TPR.txt"
    f = open(output_file, "w")
    f.write(str(TPR))
    f.close()
    #===================================================#

    equality_of_odds = {}
    equality_of_odds['after_bias_reduction'] = 0.5 * (equality_of_opportunity_y_equal_0['after_bias_reduction'] + equality_of_opportunity_y_equal_1['after_bias_reduction'])
    equality_of_odds['before_bias_reduction'] = 0.5 * (equality_of_opportunity_y_equal_0['before_bias_reduction'] + equality_of_opportunity_y_equal_1['before_bias_reduction'])
    output_file = "./output/equality_of_odds.txt"
    f = open(output_file, "w")
    f.write(str(equality_of_odds))
    f.close()

    #===================================================#

    CTF = {}
    # Load test data
    test_data = pd.read_csv('./data/'+args.dataset+'_valid_original_gender.csv')
    X_test = list(test_data["Tweets"])
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)
    
    test_data_opposite_gender = pd.read_csv('./data/'+args.dataset+'_valid_reversed_gender.csv')
    X_test_opposite_gender = list(test_data_opposite_gender["Tweets2"])
    X_test_tokenized_opposite_gender = tokenizer(X_test_opposite_gender, padding=True, truncation=True, max_length=512)
    
    # Create torch dataset
    test_dataset = Dataset(X_test_tokenized)
    # Define test trainer
    test_trainer = Trainer(model)
    # Make prediction
    raw_pred, _, _ = test_trainer.predict(test_dataset)
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)
    
    
    # Load test data for the opposite gender
    
    # Create torch dataset
    test_dataset_opposite_gender = Dataset(X_test_tokenized_opposite_gender)
    # Make prediction
    raw_pred_opposite_gender, _, _ = test_trainer.predict(test_dataset_opposite_gender)
    # Preprocess raw predictions
    y_pred_opposite_gender = np.argmax(raw_pred_opposite_gender, axis=1)
    
    demographic_parity['after_bias_reduction'] = 1-torch.abs(torch.mean(torch.from_numpy(y_pred).double()) - torch.mean(torch.from_numpy(y_pred_opposite_gender).double()))
    
    # We also compute the same metric before reducing the bias
    # Load trained model
    model_path =  "./output/checkpoint-500"
    model_before_bias_reduction = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
    
    # Define test trainer
    test_trainer_before_bias_reduction = Trainer(model_before_bias_reduction)
    # Make prediction
    raw_pred, _, _ = test_trainer_before_bias_reduction.predict(test_dataset)
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)

    # Load test data for the opposite gender
    # Create torch dataset
    test_dataset_opposite_gender = Dataset(X_test_tokenized_opposite_gender)    
    # Make prediction
    raw_pred_opposite_gender, _, _ = test_trainer_before_bias_reduction.predict(test_dataset_opposite_gender)
    # Preprocess raw predictions
    y_pred_opposite_gender = np.argmax(raw_pred_opposite_gender, axis=1)
    
    demographic_parity['before_bias_reduction'] = 1-torch.abs(torch.mean(torch.from_numpy(y_pred).double()) - torch.mean(torch.from_numpy(y_pred_opposite_gender).double()))

    output_file = "./output/demographic_parity.txt"
    f = open(output_file, "w")
    f.write(str(demographic_parity))
    f.close()

#===================================================#

    
def train_classifier(args):

    if(args.load_pretrained_classifier):
      # Load trained model
      model_name = args.classifier_model
      tokenizer = BertTokenizer.from_pretrained(model_name)
      model = BertForSequenceClassification.from_pretrained("./output/checkpoint-500", num_labels=3)
    else:
      # Define pretrained tokenizer and model
      model_name = args.classifier_model
      tokenizer = BertTokenizer.from_pretrained(model_name)
      model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

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
