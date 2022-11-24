import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from transformers import BertForSequenceClassification, RobertaForSequenceClassification
import torch
import numpy as np
import os



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softmax = torch.nn.Softmax(dim=1).to(device)


def detect_gender_words(top_attention_tokens):
    """
    This function detects whether or not the top k tokens to which the CLS token
    attends contain gender words. This is achieved by comparing the tokens
    before and after gender flipping. If they are identical, then there are no
    gender words, and vice versa.
    args:
        top_attention_tokens: a list of the top k tokens that the CLS token attends to
    returns:
        contains_gender_words: a list of boolean variables that says whether
        there are gender words among the top k tokens that the CLS attends to
        in every example.
    """
    # We log whether or not the top k attention tokens contain gender words.
    contains_gender_words = []
    for i in range(len(top_attention_tokens[0])):
        # We need to convert the attention tokens into strings to be able to do gender flipping
        top_attention_tokens_string = " ".join(
            [str(elem) for elem in top_attention_tokens[0][i]]
        )
        contains_gender_words.append(
            top_attention_tokens_string != gender_bend(top_attention_tokens_string)
        )

    return contains_gender_words


def find_biased_examples(dataset_name, data):
    """
    Find the examples that contain bias or spurious correlation by training a simple
    logisitc regression classifier and choosing the examples that the model
    classifies with very high/low p(y|x) according to some threshold. This defninition
    is also followed in this paper: https://arxiv.org/pdf/2010.02458.pdf
    args:
        dataset_name: the name of the dataset used
        data: the csv file of the dataset that we want to analyze
    returns:
        the function returns the predictions of the logistic regression classifier
    """
    data_train = pd.read_csv("./data/" + dataset_name + "_train_original_gender.csv")
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(list(data_train[data_train.columns[0]]))

    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    clf = LogisticRegression(random_state=0, solver="lbfgs", max_iter=1000).fit(
        X_train_tf, list(data_train[data_train.columns[1]])
    )

    X_counts = count_vect.transform(data[data.columns[0]])
    X_tf = tf_transformer.transform(X_counts)

    prediction_logistic_reg = clf.predict_proba(X_tf)

    return prediction_logistic_reg


def average_attention_map_all_heads(dataset, model, batch_size_debiased_model):
    """
    This function adds the attention weight maps (per example) in all the heads in the model
    args:
        dataset: the dataset for which the attention maps are computed
        model: the model used to get the attention map
        batch_size_debiased_model: the size of our batch during training the debiasing model
    returns:
        the function returns a Pytorch tensor that has the average of the attention
        map over all the heads in the model for each example
    """
    all_heads_attention_batch = []
    maximum_tokens = dataset[:]["input_ids"].shape[1]
    all_heads_attention_per_example = torch.ones(
        [0, maximum_tokens, maximum_tokens]
    ).to(device)
    for i in range(int(np.ceil(len(dataset) / batch_size_debiased_model))):
        with torch.no_grad():
            attention = model.forward(
                input_ids=torch.tensor(
                    dataset.encodings["input_ids"][
                        i * batch_size_debiased_model : (i + 1) * batch_size_debiased_model
                    ]
                ).to(device),
                attention_mask=torch.tensor(
                    dataset.encodings["attention_mask"][
                        i * batch_size_debiased_model : (i + 1) * batch_size_debiased_model
                    ]
                ).to(device),
            )["attentions"]

        attention_per_head = torch.cat(
            [torch.unsqueeze(attention[j], dim=0) for j in range(len(attention))]
        ).to(device)
        all_heads_attention_batch = torch.mean(
            attention_per_head,
            dim=[0, 2],
        )
        all_heads_attention_per_example = torch.cat(
            (all_heads_attention_per_example, all_heads_attention_batch), 0
        ).to(device)

    return all_heads_attention_per_example


def compute_confidence_and_variability(
    batch_size_biased_model,
    classifier_model,
    output_dir,
    model_dir,
    use_amulet,
    batch_size_debiased_model,
    num_epochs_biased_model,
    num_epochs_confidence_variability,
    train_dataset,
    val_dataset,
):
    """
    Compute the confidence and variability in the model before debiasing as in https://arxiv.org/pdf/2009.10795.pdf as follows:
    1) The model is trained for multple epochs, and after each epoch it is used to give predictions for a specific dataset (it could be training or validation).
    2) The examples in the dataset are categorized into "easy-to-learn", "hard-to-learn" and "ambiguous" based on the mean and standard deviation in the predictions of the ground truth labels.
    3) "easy-to-learn" examples are those that the model predicts correctly over multiple epochs, with low variability. "hard-to-learn" examples are those that the model incorrectly predicts with low variability, and "ambiguous" examples are those with high variability in the prediction.
    arguments that need explanation:
        num_epochs_confidence_variability: the number of epochs that we consider while computing confidence and variability
    returns:
        the function returns:
        confidence: the mean of the predictions that the model gives to the ground truth label, over multiple epochs.
        variability: the standard deviation of the predictions that the model gives to the ground truth label, over multiple epochs.
    """
    # Save the model weights after each epoch
    checkpoint_steps = int(train_dataset.__len__() / batch_size_biased_model)

    if classifier_model in [
        "bert-base-cased",
        "bert-large-cased",
        "distilbert-base-cased",
        "bert-base-uncased",
        "bert-large-uncased",
        "distilbert-base-uncased",
    ]:
        huggingface_model = BertForSequenceClassification
    elif classifier_model in ["roberta-base", "distilroberta-base"]:
        huggingface_model = RobertaForSequenceClassification

    prediction_epoch = []

    for k in range(num_epochs_confidence_variability):
        # Check if the model already exists
        model_checkpoint_path = (
            model_dir + "/checkpoint-" + str(checkpoint_steps * (k + 1))
        )
        if os.path.isdir(model_checkpoint_path):
            model_before_debiasing = huggingface_model.from_pretrained(
                model_checkpoint_path, num_labels=len(set(val_dataset.labels))
            ).to(device)

            number_of_labels = len(set(val_dataset.labels))
            prediction_before_debiasing = torch.ones([0, number_of_labels]).to(device)

            # Save the predictions after each epoch (based on the paper https://arxiv.org/pdf/2009.10795.pdf)
            for i in range(int(np.ceil(len(val_dataset) / batch_size_debiased_model))):
                with torch.no_grad():
                    prediction = model_before_debiasing.forward(
                        input_ids=torch.tensor(
                            val_dataset.encodings["input_ids"][
                                i * batch_size_debiased_model : (i + 1) * batch_size_debiased_model
                            ]
                        ).to(device),
                        attention_mask=torch.tensor(
                            val_dataset.encodings["attention_mask"][
                                i * batch_size_debiased_model : (i + 1) * batch_size_debiased_model
                            ]
                        ).to(device),
                    )["logits"]

                predictions_batch = softmax(
                    torch.cat(
                        [
                            torch.unsqueeze(prediction[j], dim=0)
                            for j in range(len(prediction))
                        ]
                    )
                ).to(device)
                prediction_before_debiasing = torch.cat(
                    (prediction_before_debiasing, predictions_batch), 0
                ).to(device)

            prediction_epoch.append(prediction_before_debiasing)

    ground_truth_labels = torch.tensor(val_dataset.labels).to(device)

    prediction_all = torch.cat(
        [
            torch.unsqueeze(prediction_epoch[i], dim=0)
            for i in range(len(prediction_epoch))
        ]
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


def log_topk_attention_tokens(
    batch_size_debiased_model,
    num_tokens_logged,
    data,
    model_before_debiasing,
    model_after_debiasing,
    dataset,
    tokenizer,
):
    """
    Log the top k tokens to which the classification token (CLS) attends
    returns:
        the function doesnt return anything, since the top k tokens are added to the csv file.
    """
    # Compute the attention weights in the last layer of the biased model
    top_attention_tokens_biased = []
    top_attention_tokens_weights_biased = []
    gender_tokens_biased = []
    ids = dataset[:]["input_ids"].to(device)

    all_heads_attention_before_debiasing = average_attention_map_all_heads(
        dataset, model_before_debiasing, batch_size_debiased_model
    )

    # ===================================================#

    # Log the top k tokens that the classification token attends to in the last
    # layer of the biased and de-biased models for all the heads combined
    top_attention_tokens_debiased = []
    top_attention_tokens_weights_debiased = []
    gender_tokens_debiased = []

    all_heads_attention_after_debiasing = average_attention_map_all_heads(
        dataset, model_after_debiasing, batch_size_debiased_model
    )

    # We look for 3 things:
    # 1- The top k tokens to which the CLS tokens attends.
    # 2- The values of their attention weights.
    # 3- Whether or not they refer to gender tokens.
    # This is done over all the layers and heads in the model and only for the top k tokens.
    # At the end, for every example, we have 2 values: The sum of the attention weights
    # over the top k tokens for gender words and for nongender words. We compare the distribution
    # of both values in both the biased and debiased models.
    top_attention_tokens_biased.append(
        [
            [
                tokenizer.convert_ids_to_tokens(ids[j])[i]
                for i in torch.topk(
                    all_heads_attention_before_debiasing[j][0],
                    num_tokens_logged,
                )[1]
            ]
            for j in range(len(dataset))
        ]
    )

    gender_tokens_biased.append(
        [
            [
                tokenizer.convert_ids_to_tokens(ids[j])[i]
                != gender_bend(tokenizer.convert_ids_to_tokens(ids[j])[i])
                for i in torch.topk(
                    all_heads_attention_before_debiasing[j][0],
                    num_tokens_logged,
                )[1]
            ]
            for j in range(len(dataset))
        ]
    )

    top_attention_tokens_weights_biased.append(
        [
            [
                i
                for i in torch.topk(
                    all_heads_attention_before_debiasing[j][0],
                    num_tokens_logged,
                )[0]
            ]
            for j in range(len(dataset))
        ]
    )

    attention_weights_gender_tokens_biased = torch.sum(
        torch.mul(
            torch.tensor(top_attention_tokens_weights_biased[0]),
            torch.tensor(gender_tokens_biased[0]),
        ),
        dim=1,
    )
    attention_weights_nongender_tokens_biased = torch.sum(
        torch.mul(
            torch.tensor(top_attention_tokens_weights_biased[0]),
            ~torch.tensor(gender_tokens_biased[0]),
        ),
        dim=1,
    )

    top_attention_tokens_debiased.append(
        [
            [
                tokenizer.convert_ids_to_tokens(ids[j])[i]
                for i in torch.topk(
                    all_heads_attention_after_debiasing[j][0],
                    num_tokens_logged,
                )[1]
            ]
            for j in range(len(dataset))
        ]
    )

    gender_tokens_debiased.append(
        [
            [
                tokenizer.convert_ids_to_tokens(ids[j])[i]
                != gender_bend(tokenizer.convert_ids_to_tokens(ids[j])[i])
                for i in torch.topk(
                    all_heads_attention_after_debiasing[j][0],
                    num_tokens_logged,
                )[1]
            ]
            for j in range(len(dataset))
        ]
    )

    top_attention_tokens_weights_debiased.append(
        [
            [
                i
                for i in torch.topk(
                    all_heads_attention_after_debiasing[j][0],
                    num_tokens_logged,
                )[0]
            ]
            for j in range(len(dataset))
        ]
    )

    attention_weights_gender_tokens_debiased = torch.sum(
        torch.mul(
            torch.tensor(top_attention_tokens_weights_debiased[0]),
            torch.tensor(gender_tokens_debiased[0]),
        ),
        dim=1,
    )
    attention_weights_nongender_tokens_debiased = torch.sum(
        torch.mul(
            torch.tensor(top_attention_tokens_weights_debiased[0]),
            ~torch.tensor(gender_tokens_debiased[0]),
        ),
        dim=1,
    )

    data["top attention tokens biased " + "all heads"] = top_attention_tokens_biased[0]
    data[
        "top attention tokens de-biased_" + "all_heads"
    ] = top_attention_tokens_debiased[0]

    data["gender words in top k tokens biased model"] = detect_gender_words(
        top_attention_tokens_biased
    )
    data["gender words in top k tokens de-biased model"] = detect_gender_words(
        top_attention_tokens_debiased
    )

    data[
        "Average attention weights for gender tokens biased model"
    ] = (
        attention_weights_gender_tokens_biased.cpu().numpy()
    )  # This is only over the top k tokens
    data[
        "Average attention weights for non-gender tokens biased model"
    ] = attention_weights_nongender_tokens_biased.cpu().numpy()

    data[
        "Average attention weights for gender tokens debiased model"
    ] = attention_weights_gender_tokens_debiased.cpu().numpy()
    data[
        "Average attention weights for non-gender tokens debiased model"
    ] = attention_weights_nongender_tokens_debiased.cpu().numpy()

    return data
