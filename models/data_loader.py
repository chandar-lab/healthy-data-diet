import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, encodings_gender_swap, encodings_paraphrasing, labels=None):
        self.encodings = encodings
        self.encodings_gender_swap = encodings_gender_swap
        self.encodings_paraphrasing = encodings_paraphrasing
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
    
def data_loader(args,test_subset=None,data_augmentation= None):
    """
    Load the data  from the CSV files and an object for each split in the dataset. For each dataset, we have the data stored in both the original form, as well the gender flipped form.
    args:
        args: the arguments given by the user
        test_subset: refers to whether to test on the whole test dataset, or a suset of it. The subset could be either the minority examples (the examples on which the unintended correlation helps), or the majority examples (the examples on which unintended correlation hurts.)
    returns:
        the function returns 3 objects, for the training, validation and test datasets. Each object contains the tokenized data for both genders and the labels.
    """   
    data_train = pd.read_csv("./data/" + args.dataset + "_train_original_gender.csv")
    data_valid = pd.read_csv("./data/" + args.dataset + "_valid_original_gender.csv")
    data_test = pd.read_csv("./data/" + args.dataset + "_test_original_gender.csv")    

    data_train_gender_swap = pd.read_csv("./data/" + args.dataset + "_train_gender_swap.csv")
    data_valid_gender_swap = pd.read_csv("./data/" + args.dataset + "_valid_gender_swap.csv")
    data_test_gender_swap = pd.read_csv("./data/" + args.dataset + "_test_gender_swap.csv") 

    if(test_subset == "majority"):
        data_test = data_test[data_test["majority"] == True]
        data_test_gender_swap = data_test_gender_swap[data_test_gender_swap["majority"] == True]
    elif(test_subset == "minority"):
        data_test = data_test[data_test["minority"] == True]
        data_test_gender_swap = data_test_gender_swap[data_test_gender_swap["minority"] == True]

    model_name = args.classifier_model
    tokenizer = BertTokenizer.from_pretrained(model_name)       

    # ----- 1. Preprocess data -----#
    # Preprocess data
    X_train = list(data_train[data_train.columns[0]])
    X_train_gender_swap = list(data_train_gender_swap[data_train_gender_swap.columns[0]])
    X_train_paraphrased = list(data_train[data_train.columns[4]])
    y_train = list(data_train[data_train.columns[1]])

    X_val = list(data_valid[data_valid.columns[0]])
    X_val_gender_swap = list(data_valid_gender_swap[data_valid_gender_swap.columns[0]])
    X_valid_paraphrased = list(data_valid[data_valid.columns[4]])
    y_val = list(data_valid[data_valid.columns[1]])

    X_test = list(data_test[data_test.columns[0]])
    X_test_gender_swap = list(data_test_gender_swap[data_test_gender_swap.columns[0]])
    X_test_paraphrased = list(data_test[data_test.columns[4]])
    y_test = list(data_test[data_test.columns[1]])    

    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=args.max_length)
    X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=args.max_length)
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=args.max_length)

    X_train_gender_swap_tokenized = tokenizer(X_train_gender_swap, padding=True, truncation=True, max_length=args.max_length)
    X_val_gender_swap_tokenized = tokenizer(X_val_gender_swap, padding=True, truncation=True, max_length=args.max_length)
    X_test_gender_swap_tokenized = tokenizer(X_test_gender_swap, padding=True, truncation=True, max_length=args.max_length)    

    X_train_paraphrased_tokenized = tokenizer(X_train_paraphrased, padding=True, truncation=True, max_length=args.max_length)
    X_val_paraphrased_tokenized = tokenizer(X_valid_paraphrased, padding=True, truncation=True, max_length=args.max_length)
    X_test_paraphrased_tokenized = tokenizer(X_test_paraphrased, padding=True, truncation=True, max_length=args.max_length)    

    train_dataset = Dataset(X_train_tokenized, X_train_gender_swap_tokenized, X_train_paraphrased_tokenized, y_train)
    val_dataset = Dataset(X_val_tokenized, X_val_gender_swap_tokenized, X_val_paraphrased_tokenized, y_val)
    test_dataset = Dataset(X_test_tokenized, X_test_gender_swap_tokenized, X_test_paraphrased_tokenized, y_test)
    
    return train_dataset, val_dataset, test_dataset