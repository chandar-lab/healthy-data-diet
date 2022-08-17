from transformers import BertForSequenceClassification, RobertaForSequenceClassification
import torch
import numpy as np
import os
from torch.optim import Adam
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softmax = torch.nn.Softmax(dim=1).to(device)


def compute_GraNd(
    batch_size_biased_model,
    classifier_model,
    output_dir,
    model_dir,
    use_amulet,
    num_epochs_importance_score,
    train_dataset,
):
    """
    Compute the l2 norm of the gradient of the loss function as in https://arxiv.org/pdf/2107.07075.pdf.
    The original paper suggested computing the scores in the very early stages of training, which was done on the first 20 epochs,
    given that their task needs 200 epochs for convergence. Since our tasks require 10-15 epochs for convergence, we compute the importance scores
    after 1 epoch.
    arguments that need explanation:
        num_epochs_importance_score: Number of training epochs that we consider for computing the GraNd importance scores.
    returns:
        importance scores: l2-norm of the gradient of the loss function w.r.t the weights for each input example.
    """
    checkpoint_steps = int(train_dataset.__len__() / batch_size_biased_model * num_epochs_importance_score)

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


    criterion = nn.CrossEntropyLoss()

    GraNd_epoch = []

    model_checkpoint_path = (
        model_dir + "/checkpoint-" + str(checkpoint_steps)
    )
    if os.path.isdir(model_checkpoint_path):
        model_before_debiasing = huggingface_model.from_pretrained(
            model_checkpoint_path, num_labels=len(set(train_dataset.labels))
        ).to(device)
        optimizer = Adam(model_before_debiasing.parameters())
        GraNd_all = torch.ones([0]).to(device)

        for i in range(len(train_dataset)):
            prediction = model_before_debiasing.forward(
                input_ids=torch.tensor(
                    train_dataset.encodings["input_ids"][i : (i + 1)]
                ).to(device),
                attention_mask=torch.tensor(
                    train_dataset.encodings["attention_mask"][i : (i + 1)]
                ).to(device),
            )["logits"]

            targets = torch.tensor(train_dataset.labels[i : (i + 1)]).to(device)

            loss = criterion(
                prediction.contiguous().view(-1, prediction.shape[1]),
                targets.contiguous().view(-1),
            )

            optimizer.zero_grad()
            loss.backward()

            GraNd = float(torch.tensor(0).to(device))
            for p in model_before_debiasing.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    GraNd += param_norm.item() ** 2
            GraNd = GraNd**0.5

            GraNd = torch.unsqueeze(torch.tensor(GraNd), dim=0).to(device)
            GraNd_all = torch.cat((GraNd_all, GraNd), 0).to(device)

        GraNd_epoch.append(GraNd_all)

    GraNd_all = torch.cat(
        [torch.unsqueeze(GraNd_epoch[i], dim=0) for i in range(len(GraNd_epoch))]
    )
    GraNd_mean = torch.mean(GraNd_all, dim=0).to(device)
    return GraNd_mean


def compute_GE(
    batch_size_biased_model,
    classifier_model,
    output_dir,
    model_dir,
    use_amulet,
    num_epochs_importance_score,
    batch_size_debiased_model,
    train_dataset,
):
    """
    Compute the GE score as in https://arxiv.org/pdf/2107.07075.pdf as the l2 norm of the distance between
    the prediction of the model for the original and the gender-flipped sentence. The original papers suggested computing the
    scores in the very early stages of training, which was done on the first 20 epochs, given that their task needs 200 epochs for convergence.
    Since our tasks require 10-15 epochs for convergence, we compute the importance scores after 1 epoch. Their deifnition of the importance
    score is different since they were concerned with performance, so we tailored the definition to use it for fairness.
    arguments that need explanation:
        num_epochs_importance_score: Number of training epochs that we consider for computing the GE importance scores.
    returns:
        importance scores: the score that tells us how importance each examples is for fairness experiments.
    """
    # Save the model weights after each epoch
    checkpoint_steps = int(train_dataset.__len__() / batch_size_biased_model * num_epochs_importance_score)

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


    GE_epoch = []

    model_checkpoint_path = (
        model_dir + "/checkpoint-" + str(checkpoint_steps)
    )
    if os.path.isdir(model_checkpoint_path):
        model_before_debiasing = huggingface_model.from_pretrained(
            model_checkpoint_path, num_labels=len(set(train_dataset.labels))
        ).to(device)

        number_of_labels = len(set(train_dataset.labels))
        GE_before_debiasing = torch.ones([0, number_of_labels]).to(
            device
        )

        for i in range(int(np.ceil(len(train_dataset) / batch_size_debiased_model))):
            with torch.no_grad():
                prediction_original_gender = model_before_debiasing.forward(
                    input_ids=torch.tensor(
                        train_dataset.encodings["input_ids"][
                            i * batch_size_debiased_model : (i + 1) * batch_size_debiased_model
                        ]
                    ).to(device),
                    attention_mask=torch.tensor(
                        train_dataset.encodings["attention_mask"][
                            i * batch_size_debiased_model : (i + 1) * batch_size_debiased_model
                        ]
                    ).to(device),
                )["logits"]
                prediction_gender_swap = model_before_debiasing.forward(
                    input_ids=torch.tensor(
                        train_dataset.encodings_gender_swap["input_ids"][
                            i * batch_size_debiased_model : (i + 1) * batch_size_debiased_model
                        ]
                    ).to(device),
                    attention_mask=torch.tensor(
                        train_dataset.encodings_gender_swap["attention_mask"][
                            i * batch_size_debiased_model : (i + 1) * batch_size_debiased_model
                        ]
                    ).to(device),
                )["logits"]

                GE = prediction_original_gender - prediction_gender_swap

            GE_batch = torch.cat(
                [
                    torch.unsqueeze(GE[j], dim=0)
                    for j in range(len(GE))
                ]
            ).to(device)
            GE_before_debiasing = torch.cat(
                (GE_before_debiasing, GE_batch), 0
            ).to(device)

        GE_epoch.append(GE_before_debiasing)

    GE_all = torch.cat(
        [
            torch.unsqueeze(GE_epoch[i], dim=0)
            for i in range(len(GE_epoch))
        ]
    )
    GE_mean = torch.mean(
        torch.norm(GE_all, dim=2, p=2), dim=0
    ).to(device)

    return GE_mean



def compute_forgetting_score(
    batch_size_biased_model,
    classifier_model,
    output_dir,
    model_dir,
    use_amulet,
    batch_size_debiased_model,
    train_dataset,
    num_epochs_biased_model,
):
    """
    Compute the forgettig scores as in https://arxiv.org/pdf/1812.05159.pdf as the number of times each training examples is forgotten during training.
    Forgetting happens when the example is wrongly predicted after being correctly predicted.
    returns:
        forgetting_score: the number of times each training examples is forgotten during trianing, over multiple epochs.
    """
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

    for k in range(num_epochs_biased_model):
        model_checkpoint_path = (
            model_dir + "/checkpoint-" + str(checkpoint_steps * (k + 1))
        )
        if os.path.isdir(model_checkpoint_path):
            model_before_debiasing = huggingface_model.from_pretrained(
                model_checkpoint_path, num_labels=len(set(train_dataset.labels))
            ).to(device)

            number_of_labels = len(set(train_dataset.labels))
            prediction_before_debiasing = torch.ones([0, number_of_labels]).to(device)

            for i in range(int(np.ceil(len(train_dataset) / batch_size_debiased_model))):
                with torch.no_grad():
                    prediction = model_before_debiasing.forward(
                        input_ids=torch.tensor(
                            train_dataset.encodings["input_ids"][
                                i * batch_size_debiased_model : (i + 1) * batch_size_debiased_model
                            ]
                        ).to(device),
                        attention_mask=torch.tensor(
                            train_dataset.encodings["attention_mask"][
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

    ground_truth_labels = torch.tensor(train_dataset.labels).to(device)

    prediction_all = torch.cat(
        [
            torch.unsqueeze(prediction_epoch[i], dim=0)
            for i in range(len(prediction_epoch))
        ]
    )

    y_pred_before_debiasing = torch.argmax(prediction_all, axis=2)
    correct_predictions = ground_truth_labels == y_pred_before_debiasing
    forgetting_scores = torch.sum(
        torch.logical_and(correct_predictions[:-1, :], ~correct_predictions[1:, :]),
        dim=0,
    ).to(device)

    return forgetting_scores


def compute_EL2N(
    batch_size_biased_model,
    classifier_model,
    output_dir,
    model_dir,
    use_amulet,
    num_epochs_importance_score,
    batch_size_debiased_model,
    train_dataset,
):
    """
    Compute the EL2N performance score as in https://arxiv.org/pdf/2107.07075.pdf as the l2 norm of the distance between
    the prediction of the model and groundtruth label for every examples. The original papers suggested computing the
    scores in the very early stages of training, which was done on the first 20 epochs, given that their task needs 200 epochs for convergence.
    Since our tasks require 10-15 epochs for convergence, we compute the importance scores after 1 epoch.
    arguments that need explanation:
        num_epochs_importance_score: Number of training epochs that we consider for computing the El2N importance scores.
    returns:
        El2N importance scores: the score that tells us how importance each examples is for performance experiments.
    """
    checkpoint_steps = int(train_dataset.__len__() / batch_size_biased_model * num_epochs_importance_score)

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

    EL2N_epoch = []

    model_checkpoint_path = (
        model_dir + "/checkpoint-" + str(checkpoint_steps))
    
    if os.path.isdir(model_checkpoint_path):
        model_before_debiasing = huggingface_model.from_pretrained(
            model_checkpoint_path, num_labels=len(set(train_dataset.labels))
        ).to(device)

        number_of_labels = len(set(train_dataset.labels))
        EL2N_before_debiasing = torch.ones([0, number_of_labels]).to(
            device
        )

        for i in range(int(np.ceil(len(train_dataset) / batch_size_debiased_model))):
            with torch.no_grad():
                prediction_original_gender = model_before_debiasing.forward(
                    input_ids=torch.tensor(
                        train_dataset.encodings["input_ids"][
                            i * batch_size_debiased_model : (i + 1) * batch_size_debiased_model
                        ]
                    ).to(device),
                    attention_mask=torch.tensor(
                        train_dataset.encodings["attention_mask"][
                            i * batch_size_debiased_model : (i + 1) * batch_size_debiased_model
                        ]
                    ).to(device),
                )["logits"]

                EL2N = softmax(prediction_original_gender).to(
                    device
                ) - torch.nn.functional.one_hot(
                    torch.tensor(
                        train_dataset.labels[i * batch_size_debiased_model : (i + 1) * batch_size_debiased_model]
                    ),
                    num_classes=len(set(train_dataset.labels)),
                ).to(
                    device
                )

            EL2N_batch = torch.cat(
                [
                    torch.unsqueeze(EL2N[j], dim=0)
                    for j in range(len(EL2N))
                ]
            ).to(device)
            EL2N_before_debiasing = torch.cat(
                (EL2N_before_debiasing, EL2N_batch), 0
            ).to(device)

        EL2N_epoch.append(EL2N_before_debiasing)

    EL2N_all = torch.cat(
        [
            torch.unsqueeze(EL2N_epoch[i], dim=0)
            for i in range(len(EL2N_epoch))
        ]
    )
    EL2N_mean = torch.mean(
        torch.norm(EL2N_all, dim=2, p=2), dim=0
    ).to(device)

    return EL2N_mean
