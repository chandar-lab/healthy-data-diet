import pandas as pd
import random
from transformers import BertTokenizer, RobertaTokenizer
import torch
import numpy as np


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        encodings,
        encodings_gender_swap=None,
        encodings_gender_blind=None,
        gender_swap=None,
        labels=None,
    ):
        self.encodings = encodings
        self.encodings_gender_swap = encodings_gender_swap
        self.encodings_gender_blind = encodings_gender_blind
        self.gender_swap = gender_swap
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def data_loader(
    dataset,
    CDA_examples_ranking,
    data_augmentation_ratio,
    data_diet_examples_ranking,
    data_diet_factual_ratio,
    data_diet_counterfactual_ratio,
    data_substitution_ratio,
    max_length,
    classifier_model,
    apply_data_augmentation=None,
    apply_data_substitution=None,
    apply_blindness=None,
    apply_data_diet=None,
    apply_data_balancing=None,
    IPTTS=False,
):
    """
    Load the data  from the CSV files and an object for each split in the dataset.
    For each dataset, we have the data stored in both the original form, as well
    the gender flipped form.
    args:
        dataset: the dataset used
        CDA_examples_ranking: the ranking of the CDa examples
        data_augmentation_ratio: The ratio of data augmentation that we apply, given that the debiasing is using data augmentation
        data_diet_examples_ranking: Type of rankings we use to pick up the examples in data pruning.
        data_diet_factual_ratio: The ratio of the factual examples that we train on while using data diet.
        data_diet_counterfactual_ratio: The ratio of the counterfactual examples that we train on while using data diet.
        data_substitution_ratio: The ratio of the dataset examples that are flipped in data substitution.
        max_length: The maximum length of the sentences that we classify (in terms of the number of tokens)
        classifier_model: the model name
        apply_data_augmentation: a flag to choose whether or not to apply data
            augmentation, meaning that the number of examples doubles because we flip
            the gender in each example and add it as a new example.
            apply_data_diet: a flag to choose whether or not to apply data
            diet, meaning that prune the examples.
        apply_data_substitution: a flag to choose whether or not to apply data
            substitution, meaning that with a probability of 0.5 we flip the gender
            in each example https://arxiv.org/abs/1909.00871.
        apply_blindness: a flag to choose whether or not to apply data
            blindness, meaning that we remove all the gender-related words, such as
            names and pronouns.
        IPTTS: a flag that means that we want to also return the IPTTS dataset,
            which is the dataset on which we compute the bias metrics. We can also
            compute the AUC score on it.
    returns:
        the function returns 3 objects, for the training, validation and test
        datasets. Each object contains the tokenized data and the corresponding
        labels.
    """
    data_train = pd.read_csv("./data/" + dataset + "_train_original_gender.csv")
    data_valid = pd.read_csv("./data/" + dataset + "_valid_original_gender.csv")
    data_test = pd.read_csv("./data/" + dataset + "_test_original_gender.csv")

    # The gender swap means that we flip the gender in each example in out dataset.
    # For example, the sentence "he is a doctor" becomes "she is a doctor".
    data_train_gender_swap = pd.read_csv("./data/" + dataset + "_train_gender_swap.csv")
    # The gender blind means that we remove the gender in each example in out dataset.
    # For example, the sentence "he is a doctor" becomes "is a doctor".
    data_train_gender_blind = pd.read_csv(
        "./data/" + dataset + "_train_gender_blind.csv"
    )

    if dataset == "Wiki":
        # "Balancing" means that we the number of examples that refer to both different genders is the same.
        # This baseline is only iplemented for the Wikipedia dataset, following Dixon et al. paper https://storage.googleapis.com/pub-tools-public-publication-data/pdf/ab50a4205513d19233233dbdbb4d1035d7c8c6c2.pdf
        data_train_balanced = pd.read_csv("./data/" + dataset + "_train_balanced.csv")

    # This is a boolean tensor that indentifies the examples that have undergone gender swapping
    train_gender_swap = torch.tensor(
        data_train[data_train.columns[0]]
        != data_train_gender_swap[data_train_gender_swap.columns[0]]
    )

    if apply_blindness:
        data_train = data_train_gender_blind

    if apply_data_balancing:
        data_train = data_train_balanced

    if apply_data_augmentation:

        # In data augmentation, we double the size of the training data by adding
        # the gender-fliped example of every training example.
        number_of_original_examples = len(data_train_gender_swap)

        # We now duplicate the examples. We make sure that the column names are the same,
        # to be able to concatenate them into a single data frame
        data_train_gender_swap = data_train_gender_swap.rename(
            columns={data_train_gender_swap.columns[0]: data_train.columns[0]}
        )

        if CDA_examples_ranking != "random":

            # This is the threshold for the most important k percentile in the dataset,
            # such tat any exmaple that has a score higher than this threshold
            # is considered an important example. The idea is to get close to
            # the performance of data augmentation, but with less exampels.
            if data_augmentation_ratio != 0:
                # If we want to add 30% more examples, we add the top 30% examples after sorting them according to the importance score.
                # We assume that the distribution of importance scores is uniform so that it becomes easier.
                idx_important_examples = (
                    data_train[CDA_examples_ranking]
                    .sort_values(ascending=False)
                    .iloc[0 : int(len(data_train) * data_augmentation_ratio)]
                    .index
                )

                data_train = pd.concat(
                    [data_train, data_train_gender_swap.iloc[idx_important_examples]],
                    axis=0,
                    ignore_index=True,
                )

                data_train_gender_swap = pd.concat(
                    [data_train_gender_swap, data_train.iloc[idx_important_examples]],
                    axis=0,
                    ignore_index=True,
                )
                data_train_gender_blind = pd.concat(
                    [
                        data_train_gender_blind,
                        data_train_gender_blind.iloc[idx_important_examples],
                    ],
                    axis=0,
                    ignore_index=True,
                )
        else:

            data_train = pd.concat(
                [
                    data_train,
                    data_train_gender_swap.iloc[
                        0 : int(data_augmentation_ratio * number_of_original_examples)
                    ],
                ],
                axis=0,
                ignore_index=True,
            )
            # We also do data augmentation for the gender-swapped examples.
            data_train_gender_swap = pd.concat(
                [
                    data_train_gender_swap,
                    data_train.iloc[
                        0 : int(data_augmentation_ratio * number_of_original_examples)
                    ],
                ],
                axis=0,
                ignore_index=True,
            )
            # We also do data augmentation for the gender blind examples.
            data_train_gender_blind = pd.concat(
                [
                    data_train_gender_blind,
                    data_train_gender_blind.iloc[
                        0 : int(data_augmentation_ratio * number_of_original_examples)
                    ],
                ],
                axis=0,
                ignore_index=True,
            )

    if apply_data_diet:
        # In data diet, we prune the examples to achieve the desired performance and fairness with the least number of training examples.
        data_train_gender_swap = data_train_gender_swap.rename(
            columns={data_train_gender_swap.columns[0]: data_train.columns[0]}
        )

        data_train_important_performance = pd.DataFrame()
        data_train_important_fairness = pd.DataFrame()
        data_train_gender_swap_important_performance = pd.DataFrame()
        data_train_gender_swap_important_fairness = pd.DataFrame()

        idx_important_examples_performance = []
        idx_important_examples_fairness = []
        number_of_original_examples = len(data_train_gender_swap)
        if data_diet_examples_ranking == "healthy_diet":
            # If we are doing data pruning using our "healthy diet" method, we choose the examples that are important for the performance based on https://arxiv.org/pdf/2107.07075.pdf,
            # while the examples important for fairness based on our EL2N for fariness metric. The "EL2N" data pruning will choose both based on the paper https://arxiv.org/pdf/2107.07075.pdf.
            performance_ranking = "forgetting_scores"
            fairness_ranking = "El2N_fairness"
            idx_important_examples_performance = (
                data_train[performance_ranking]
                .sort_values(ascending=False)
                .iloc[0 : int(len(data_train) * data_diet_factual_ratio)]
                .index
            )
            idx_important_examples_fairness = (
                data_train[fairness_ranking]
                .sort_values(ascending=False)
                .iloc[0 : int(len(data_train) * data_diet_counterfactual_ratio)]
                .index
            )

        if data_diet_examples_ranking == "fairness_only_diet":
            # If we are doing data pruning using our "healthy diet" method, we choose the examples that are important for the performance based on https://arxiv.org/pdf/2107.07075.pdf,
            # while the examples important for fairness based on our EL2N for fariness metric. The "EL2N" data pruning will choose both based on the paper https://arxiv.org/pdf/2107.07075.pdf.
            performance_ranking = "El2N_fairness"
            fairness_ranking = "El2N_fairness"

            idx_important_examples_performance = (
                data_train[performance_ranking]
                .sort_values(ascending=False)
                .iloc[0 : int(len(data_train) * data_diet_factual_ratio)]
                .index
            )
            idx_important_examples_fairness = (
                data_train[fairness_ranking]
                .sort_values(ascending=False)
                .iloc[0 : int(len(data_train) * data_diet_counterfactual_ratio)]
                .index
            )

        elif data_diet_examples_ranking == "El2N":

            performance_ranking = "El2N_performance"
            idx_important_examples_performance = (
                data_train[performance_ranking]
                .sort_values(ascending=False)
                .iloc[0 : int(len(data_train) * data_diet_factual_ratio)]
                .index
            )
            idx_important_examples_fairness = [
                random.randrange(0, number_of_original_examples)
                for i in range(int(len(data_train) * data_diet_counterfactual_ratio))
            ]

        elif data_diet_examples_ranking == "forgetting_scores":

            performance_ranking = "forgetting_scores"
            idx_important_examples_performance = (
                data_train[performance_ranking]
                .sort_values(ascending=False)
                .iloc[0 : int(len(data_train) * data_diet_factual_ratio)]
                .index
            )
            idx_important_examples_fairness = [
                random.randrange(0, number_of_original_examples)
                for i in range(int(len(data_train) * data_diet_counterfactual_ratio))
            ]

        elif data_diet_examples_ranking == "GradN":

            performance_ranking = "GradN"
            idx_important_examples_performance = (
                data_train[performance_ranking]
                .sort_values(ascending=False)
                .iloc[0 : int(len(data_train) * data_diet_factual_ratio)]
                .index
            )
            idx_important_examples_fairness = [
                random.randrange(0, number_of_original_examples)
                for i in range(int(len(data_train) * data_diet_counterfactual_ratio))
            ]

        elif data_diet_examples_ranking == "random":

            idx_important_examples_performance += [
                random.randrange(0, number_of_original_examples)
                for i in range(int(len(data_train) * data_diet_factual_ratio))
            ]
            idx_important_examples_fairness = [
                random.randrange(0, number_of_original_examples)
                for i in range(int(len(data_train) * data_diet_counterfactual_ratio))
            ]

        data_train_important_performance = data_train.iloc[
            idx_important_examples_performance
        ]
        data_train_gender_swap_important_performance = data_train_gender_swap.iloc[
            idx_important_examples_performance
        ]
        data_train_important_fairness = data_train_gender_swap.iloc[
            idx_important_examples_fairness
        ]
        data_train_gender_swap_important_fairness = data_train.iloc[
            idx_important_examples_fairness
        ]

        data_train = pd.concat(
            [data_train_important_performance, data_train_important_fairness],
            axis=0,
            ignore_index=True,
        )

        data_train_gender_swap = pd.concat(
            [
                data_train_gender_swap_important_performance,
                data_train_gender_swap_important_fairness,
            ],
            axis=0,
            ignore_index=True,
        )

    if apply_data_substitution:
        # We substitute each example with the gender-flipped one, with a probability of 0.5,
        # as described in https://arxiv.org/abs/1909.00871
        for i in range(len(data_train)):
            if np.random.uniform(0, 1, 1)[0] < data_substitution_ratio:
                temp = data_train_gender_swap[data_train_gender_swap.columns[0]].iloc[i]
                data_train_gender_swap[data_train_gender_swap.columns[0]].iloc[
                    i
                ] = data_train[data_train.columns[0]].iloc[i]
                data_train[data_train.columns[0]].iloc[i] = temp
    model_name = classifier_model
    if model_name in [
        "bert-base-cased",
        "bert-large-cased",
        "distilbert-base-cased",
        "bert-base-uncased",
        "bert-large-uncased",
        "distilbert-base-uncased",
    ]:
        tokenizer = BertTokenizer.from_pretrained(
            "./saved_models/cached_tokenizers/" + model_name
        )
    elif model_name in ["roberta-base", "distilroberta-base"]:
        tokenizer = RobertaTokenizer.from_pretrained(
            "./saved_models/cached_tokenizers/" + model_name
        )

    # ----- 1. Preprocess data -----#
    # Preprocess data
    X_train = list(data_train[data_train.columns[0]])
    y_train = list(data_train[data_train.columns[1]])

    X_val = list(data_valid[data_valid.columns[0]])
    y_val = list(data_valid[data_valid.columns[1]])

    X_test = list(data_test[data_test.columns[0]])
    y_test = list(data_test[data_test.columns[1]])

    X_train_gender_swap = list(
        data_train_gender_swap[data_train_gender_swap.columns[0]]
    )

    X_train_gender_swap_tokenized = tokenizer(
        X_train_gender_swap, padding=True, truncation=True, max_length=max_length
    )

    X_train_gender_blind = list(
        data_train_gender_blind[data_train_gender_blind.columns[0]]
    )

    X_train_gender_blind_tokenized = tokenizer(
        X_train_gender_blind, padding=True, truncation=True, max_length=max_length
    )

    X_train_tokenized = tokenizer(
        X_train, padding=True, truncation=True, max_length=max_length
    )
    X_val_tokenized = tokenizer(
        X_val, padding=True, truncation=True, max_length=max_length
    )
    X_test_tokenized = tokenizer(
        X_test, padding=True, truncation=True, max_length=max_length
    )

    train_dataset = Dataset(
        encodings=X_train_tokenized,
        encodings_gender_swap=X_train_gender_swap_tokenized,
        encodings_gender_blind=X_train_gender_blind_tokenized,
        gender_swap=train_gender_swap,
        labels=y_train,
    )
    val_dataset = Dataset(
        encodings=X_val_tokenized,
        labels=y_val,
    )
    test_dataset = Dataset(
        encodings=X_test_tokenized,
        labels=y_test,
    )

    # IPTTS is a synthetic dataset that is used to compute the fairness metrics
    if IPTTS:
        data_IPTTS_gender = pd.read_csv("./data/" + "madlib.csv")
        X_IPTTS_gender = list(data_IPTTS_gender[data_IPTTS_gender.columns[0]])
        y_IPTTS_gender = list(data_IPTTS_gender["Class"])
        X_IPTTS_gender_tokenized = tokenizer(
            X_IPTTS_gender, padding=True, truncation=True, max_length=max_length
        )
        IPTTS_gender_dataset = Dataset(
            encodings=X_IPTTS_gender_tokenized,
            labels=y_IPTTS_gender,
        )

        data_IPTTS_social = pd.read_csv("./data/" + "bias_madlibs_77k.csv")
        X_IPTTS_social = list(data_IPTTS_social[data_IPTTS_social.columns[0]])
        y_IPTTS_social = list(data_IPTTS_social["Class"])
        X_IPTTS_social_tokenized = tokenizer(
            X_IPTTS_social, padding=True, truncation=True, max_length=max_length
        )
        IPTTS_social_dataset = Dataset(
            encodings=X_IPTTS_social_tokenized,
            labels=y_IPTTS_social,
        )

        return (
            train_dataset,
            val_dataset,
            test_dataset,
            IPTTS_gender_dataset,
            IPTTS_social_dataset,
        )

    return train_dataset, val_dataset, test_dataset
