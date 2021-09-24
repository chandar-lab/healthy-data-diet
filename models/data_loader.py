import pandas as pd
from transformers import BertTokenizer
import torch

# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        encodings,
        encodings_gender_swap=None,
        encodings_paraphrasing=None,
        gender_swap=None,
        labels=None,
    ):
        self.encodings = encodings
        self.encodings_gender_swap = encodings_gender_swap
        self.encodings_paraphrasing = encodings_paraphrasing
        self.gender_swap = gender_swap
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def data_loader(args, subset=None, apply_data_augmentation=None, IPTTS=False):
    """
    Load the data  from the CSV files and an object for each split in the dataset.
    For each dataset, we have the data stored in both the original form, as well
    the gender flipped form.
    args:
        args: the arguments given by the user
        subset: refers to whether to consider the whole dataset or a suset of it.
        The subset could be either the minority examples (the examples on which
        the unintended correlation helps), or the majority examples (the examples
        on which unintended correlation hurts).
        apply_data_augmentation: a flag to choose whether or not to apply data
        augmentation, meaning that the number of examples doubles because we flip
        the gender in each example and add it as a new example.
        IPTTS: a flag that means that we want to also return the IPTTS dataset,
        which is the dataset on which we compute the bias metrics. We can also
        compute the AUC score on it.
    returns:
        the function returns 3 objects, for the training, validation and test
        datasets. Each object contains the tokenized data for both genders and
        the labels.
    """
    data_train = pd.read_csv("./data/" + args.dataset + "_train_original_gender.csv")
    data_valid = pd.read_csv("./data/" + args.dataset + "_valid_original_gender.csv")
    data_test = pd.read_csv("./data/" + args.dataset + "_test_original_gender.csv")

    # The gender swap means that we flip the gender in each example in out dataset.
    # For example, the sentence "he is a doctor" becomes "she is a doctor".
    data_train_gender_swap = pd.read_csv(
        "./data/" + args.dataset + "_train_gender_swap.csv"
    )
    data_valid_gender_swap = pd.read_csv(
        "./data/" + args.dataset + "_valid_gender_swap.csv"
    )
    data_test_gender_swap = pd.read_csv(
        "./data/" + args.dataset + "_test_gender_swap.csv"
    )

    # This is a boolean tensor that indentifies the examples that have undergone gender swapping
    train_gender_swap = torch.tensor(
        data_train[data_train.columns[0]]
        != data_train_gender_swap[data_train_gender_swap.columns[0]]
    )
    valid_gender_swap = torch.tensor(
        data_valid[data_valid.columns[0]]
        != data_valid_gender_swap[data_valid_gender_swap.columns[0]]
    )
    test_gender_swap = torch.tensor(
        data_test[data_test.columns[0]]
        != data_test_gender_swap[data_test_gender_swap.columns[0]]
    )

    # The paraphrasing means that we each sentence in a different way, while
    # preserving the meaning. For example, the sentence "I really liked the movie"
    # becomes "I enjoyed the movie".
    X_train_paraphrased = list(data_train["data augmentation"])
    if apply_data_augmentation == True:
        # We make sure that the column names are the same, to be able to concatenate them into a single data frame
        data_train_gender_swap = data_train_gender_swap.rename(
            columns={data_train_gender_swap.columns[0]: data_train.columns[0]}
        )
        data_train = pd.concat(
            [data_train, data_train_gender_swap], axis=0, ignore_index=True
        )

    if subset == "majority":
        data_test = data_test[data_test["majority"] == True]
        data_test_gender_swap = data_test_gender_swap[
            data_test_gender_swap["majority"] == True
        ]

        data_train = data_train[data_train["majority"] == True]
        data_train_gender_swap = data_train_gender_swap[
            data_train_gender_swap["majority"] == True
        ]

        data_valid = data_valid[data_valid["majority"] == True]
        data_valid_gender_swap = data_valid_gender_swap[
            data_valid_gender_swap["majority"] == True
        ]
    elif subset == "minority":
        data_test = data_test[data_test["minority"] == True]
        data_test_gender_swap = data_test_gender_swap[
            data_test_gender_swap["minority"] == True
        ]

        data_train = data_train[data_train["minority"] == True]
        data_train_gender_swap = data_train_gender_swap[
            data_train_gender_swap["minority"] == True
        ]

        data_valid = data_valid[data_valid["minority"] == True]
        data_valid_gender_swap = data_valid_gender_swap[
            data_valid_gender_swap["minority"] == True
        ]

    model_name = args.classifier_model
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # ----- 1. Preprocess data -----#
    # Preprocess data
    X_train = list(data_train[data_train.columns[0]])
    X_train_gender_swap = list(
        data_train_gender_swap[data_train_gender_swap.columns[0]]
    )
    y_train = list(data_train[data_train.columns[1]])

    X_val = list(data_valid[data_valid.columns[0]])
    X_val_gender_swap = list(data_valid_gender_swap[data_valid_gender_swap.columns[0]])
    X_valid_paraphrased = list(data_valid["data augmentation"])
    y_val = list(data_valid[data_valid.columns[1]])

    X_test = list(data_test[data_test.columns[0]])
    X_test_gender_swap = list(data_test_gender_swap[data_test_gender_swap.columns[0]])
    X_test_paraphrased = list(data_test["data augmentation"])
    y_test = list(data_test[data_test.columns[1]])

    X_train_tokenized = tokenizer(
        X_train, padding=True, truncation=True, max_length=args.max_length
    )
    X_val_tokenized = tokenizer(
        X_val, padding=True, truncation=True, max_length=args.max_length
    )
    X_test_tokenized = tokenizer(
        X_test, padding=True, truncation=True, max_length=args.max_length
    )

    X_train_gender_swap_tokenized = tokenizer(
        X_train_gender_swap, padding=True, truncation=True, max_length=args.max_length
    )
    X_val_gender_swap_tokenized = tokenizer(
        X_val_gender_swap, padding=True, truncation=True, max_length=args.max_length
    )
    X_test_gender_swap_tokenized = tokenizer(
        X_test_gender_swap, padding=True, truncation=True, max_length=args.max_length
    )

    X_train_paraphrased_tokenized = tokenizer(
        X_train_paraphrased, padding=True, truncation=True, max_length=args.max_length
    )
    X_val_paraphrased_tokenized = tokenizer(
        X_valid_paraphrased, padding=True, truncation=True, max_length=args.max_length
    )
    X_test_paraphrased_tokenized = tokenizer(
        X_test_paraphrased, padding=True, truncation=True, max_length=args.max_length
    )

    train_dataset = Dataset(
        X_train_tokenized,
        X_train_gender_swap_tokenized,
        X_train_paraphrased_tokenized,
        train_gender_swap,
        y_train,
    )
    val_dataset = Dataset(
        X_val_tokenized,
        X_val_gender_swap_tokenized,
        X_val_paraphrased_tokenized,
        valid_gender_swap,
        y_val,
    )
    test_dataset = Dataset(
        X_test_tokenized,
        X_test_gender_swap_tokenized,
        X_test_paraphrased_tokenized,
        test_gender_swap,
        y_test,
    )

    if IPTTS == True and args.dataset == "Twitter_sexism_dataset":
        data_IPTTS = pd.read_csv("./data/" + "madlib.csv")
        X_IPTTS = list(data_IPTTS[data_IPTTS.columns[0]])
        y_IPTTS = list(data_IPTTS["Class"])
        X_IPTTS_tokenized = tokenizer(
            X_IPTTS, padding=True, truncation=True, max_length=args.max_length
        )
        IPTTS_dataset = Dataset(
            encodings=X_IPTTS_tokenized,
            labels=y_IPTTS,
        )

        return train_dataset, val_dataset, test_dataset, IPTTS_dataset

    return train_dataset, val_dataset, test_dataset
