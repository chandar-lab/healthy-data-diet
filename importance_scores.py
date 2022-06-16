from transformers import BertForSequenceClassification, RobertaForSequenceClassification
import torch
import numpy as np
import os
from torch.optim import Adam
import torch.nn as nn

# from gender_bender import gender_bend
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softmax = torch.nn.Softmax(dim=1).to(device)


def compute_GradN(
    batch_size_pretraining,
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
    given that their task needs 200 epochs for convergence. Since our task requires 10 epochs for convergence, we compute the importance scores
    after 1 epoch.
    arguments that need explanation:
        num_epochs_importance_score: Number of training epochs that we consider for computing the El2N and GradN importance scores.
    returns:
        importance scores: l2-norm of the gradient of the loss function w.r.t the weights for each input example.
    """
    # Save the model weights after each epoch
    checkpoint_steps = int(train_dataset.__len__() / batch_size_pretraining)

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

    GradN_epoch = []

    for k in range(num_epochs_importance_score):
        # Check if the model already exists
        # todo: The score is computed here over few epochs. I need to double check that this is how it should be. Since, I am using num_epochs_importance_score = 1, it doesn't matter for now
        model_checkpoint_path = (
            model_dir + "/checkpoint-" + str(checkpoint_steps * (k + 1))
        )
        if os.path.isdir(model_checkpoint_path):
            model_before_debiasing = huggingface_model.from_pretrained(
                model_checkpoint_path, num_labels=len(set(train_dataset.labels))
            ).to(device)
            optimizer = Adam(model_before_debiasing.parameters())
            GradN_all = torch.ones([0]).to(device)

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

                GradN = float(torch.tensor(0).to(device))
                for p in model_before_debiasing.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        GradN += param_norm.item() ** 2
                GradN = GradN**0.5

                GradN = torch.unsqueeze(torch.tensor(GradN), dim=0).to(device)
                GradN_all = torch.cat((GradN_all, GradN), 0).to(device)

            GradN_epoch.append(GradN_all)

    GradN_all = torch.cat(
        [torch.unsqueeze(GradN_epoch[i], dim=0) for i in range(len(GradN_epoch))]
    )
    GradN_mean = torch.mean(GradN_all, dim=0).to(device)
    return GradN_mean


def compute_El2N_fairness(
    batch_size_pretraining,
    classifier_model,
    output_dir,
    model_dir,
    use_amulet,
    num_epochs_importance_score,
    batch_size,
    train_dataset,
):
    """
    Compute the EL2N fairness score as in https://arxiv.org/pdf/2107.07075.pdf as the l2 norm of the distance between
    the prediction of the model for the original and the gender-flipped sentence. The original papers suggested computing the
    scores in the very early stages of training, which was done on the first 20 epochs, given that their task needs 200 epochs for convergence.
    Since our task requires 10 epochs for convergence, we compute the importance scores after 1 epoch. Their deifnition of the importance
    score is different since they were concerned with performance, so we tailored the definition to use it for fairness.
    arguments that need explanation:
        num_epochs_importance_score: Number of training epochs that we consider for computing the El2N and GradN importance scores.
    returns:
        importance scores: the score that tells us how importance each examples is for fairness experiments.
    """
    # Save the model weights after each epoch
    checkpoint_steps = int(train_dataset.__len__() / batch_size_pretraining * num_epochs_importance_score)

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


    El2N_fairness_epoch = []

    #for k in range(num_epochs_importance_score):
        # Check if the model already exists
    model_checkpoint_path = (
        model_dir + "/checkpoint-" + str(checkpoint_steps)
    )
    if os.path.isdir(model_checkpoint_path):
        model_before_debiasing = huggingface_model.from_pretrained(
            model_checkpoint_path, num_labels=len(set(train_dataset.labels))
        ).to(device)

        number_of_labels = len(set(train_dataset.labels))
        El2N_fairness_before_debiasing = torch.ones([0, number_of_labels]).to(
            device
        )

        # Save the predictions after each epoch (based on the paper https://arxiv.org/pdf/2009.10795.pdf)
        for i in range(int(np.ceil(len(train_dataset) / batch_size))):
            with torch.no_grad():
                prediction_original_gender = model_before_debiasing.forward(
                    input_ids=torch.tensor(
                        train_dataset.encodings["input_ids"][
                            i * batch_size : (i + 1) * batch_size
                        ]
                    ).to(device),
                    attention_mask=torch.tensor(
                        train_dataset.encodings["attention_mask"][
                            i * batch_size : (i + 1) * batch_size
                        ]
                    ).to(device),
                )["logits"]
                prediction_gender_swap = model_before_debiasing.forward(
                    input_ids=torch.tensor(
                        train_dataset.encodings_gender_swap["input_ids"][
                            i * batch_size : (i + 1) * batch_size
                        ]
                    ).to(device),
                    attention_mask=torch.tensor(
                        train_dataset.encodings_gender_swap["attention_mask"][
                            i * batch_size : (i + 1) * batch_size
                        ]
                    ).to(device),
                )["logits"]

                El2N_fairness = prediction_original_gender - prediction_gender_swap

            El2N_fairness_batch = torch.cat(
                [
                    torch.unsqueeze(El2N_fairness[j], dim=0)
                    for j in range(len(El2N_fairness))
                ]
            ).to(device)
            El2N_fairness_before_debiasing = torch.cat(
                (El2N_fairness_before_debiasing, El2N_fairness_batch), 0
            ).to(device)

        El2N_fairness_epoch.append(El2N_fairness_before_debiasing)

    El2N_fairness_all = torch.cat(
        [
            torch.unsqueeze(El2N_fairness_epoch[i], dim=0)
            for i in range(len(El2N_fairness_epoch))
        ]
    )
    El2N_fairness_mean = torch.mean(
        torch.norm(El2N_fairness_all, dim=2, p=2), dim=0
    ).to(device)

    return El2N_fairness_mean


def compute_forgetting_score(
    batch_size_pretraining,
    classifier_model,
    output_dir,
    model_dir,
    use_amulet,
    batch_size,
    train_dataset,
    num_epochs_pretraining,
):
    """
    Compute the forgettig scores as in https://arxiv.org/pdf/1812.05159.pdf as the number of times each training examples is forgotten during training.
    Forgetting happens when the example is wrongly predicted after being correctly predicted.
    returns:
        forgetting_score: the number of times each training examples is forgotten during trianing, over multiple epochs.
    """
    # Save the model weights after each epoch
    checkpoint_steps = int(train_dataset.__len__() / batch_size_pretraining)

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

    for k in range(num_epochs_pretraining):
        # Check if the model already exists
        model_checkpoint_path = (
            model_dir + "/checkpoint-" + str(checkpoint_steps * (k + 1))
        )
        if os.path.isdir(model_checkpoint_path):
            model_before_debiasing = huggingface_model.from_pretrained(
                model_checkpoint_path, num_labels=len(set(train_dataset.labels))
            ).to(device)

            number_of_labels = len(set(train_dataset.labels))
            prediction_before_debiasing = torch.ones([0, number_of_labels]).to(device)

            # Save the predictions after each epoch (based on the paper https://arxiv.org/pdf/2009.10795.pdf)
            for i in range(int(np.ceil(len(train_dataset) / batch_size))):
                with torch.no_grad():
                    prediction = model_before_debiasing.forward(
                        input_ids=torch.tensor(
                            train_dataset.encodings["input_ids"][
                                i * batch_size : (i + 1) * batch_size
                            ]
                        ).to(device),
                        attention_mask=torch.tensor(
                            train_dataset.encodings["attention_mask"][
                                i * batch_size : (i + 1) * batch_size
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


def compute_El2N_performance(
    batch_size_pretraining,
    classifier_model,
    output_dir,
    model_dir,
    use_amulet,
    num_epochs_importance_score,
    batch_size,
    train_dataset,
):
    """
    Compute the EL2N performance score as in https://arxiv.org/pdf/2107.07075.pdf as the l2 norm of the distance between
    the prediction of the model and groundtruth label for every examples. The original papers suggested computing the
    scores in the very early stages of training, which was done on the first 20 epochs, given that their task needs 200 epochs for convergence.
    Since our task requires 10 epochs for convergence, we compute the importance scores after 1 epoch.
    arguments that need explanation:
        num_epochs_importance_score: Number of training epochs that we consider for computing the El2N and GradN importance scores.
    returns:
        El2N importance scores: the score that tells us how importance each examples is for performance experiments.
    """
    # Save the model weights after each epoch
    checkpoint_steps = int(train_dataset.__len__() / batch_size_pretraining)

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

    El2N_performance_epoch = []

    for k in range(num_epochs_importance_score):
        # Check if the model already exists
        # todo: The score is computed here over few epochs. I need to double check that this is how it should be. Since, I am using num_epochs_importance_score = 1, it doesn't matter for now
        model_checkpoint_path = (
            model_dir + "/checkpoint-" + str(checkpoint_steps * (k + 1))
        )
        if os.path.isdir(model_checkpoint_path):
            model_before_debiasing = huggingface_model.from_pretrained(
                model_checkpoint_path, num_labels=len(set(train_dataset.labels))
            ).to(device)

            number_of_labels = len(set(train_dataset.labels))
            El2N_performance_before_debiasing = torch.ones([0, number_of_labels]).to(
                device
            )

            # Save the predictions after each epoch (based on the paper https://arxiv.org/pdf/2009.10795.pdf)
            for i in range(int(np.ceil(len(train_dataset) / batch_size))):
                with torch.no_grad():
                    prediction_original_gender = model_before_debiasing.forward(
                        input_ids=torch.tensor(
                            train_dataset.encodings["input_ids"][
                                i * batch_size : (i + 1) * batch_size
                            ]
                        ).to(device),
                        attention_mask=torch.tensor(
                            train_dataset.encodings["attention_mask"][
                                i * batch_size : (i + 1) * batch_size
                            ]
                        ).to(device),
                    )["logits"]

                    El2N_performance = softmax(prediction_original_gender).to(
                        device
                    ) - torch.nn.functional.one_hot(
                        torch.tensor(
                            train_dataset.labels[i * batch_size : (i + 1) * batch_size]
                        ),
                        num_classes=len(set(train_dataset.labels)),
                    ).to(
                        device
                    )

                El2N_performance_batch = torch.cat(
                    [
                        torch.unsqueeze(El2N_performance[j], dim=0)
                        for j in range(len(El2N_performance))
                    ]
                ).to(device)
                El2N_performance_before_debiasing = torch.cat(
                    (El2N_performance_before_debiasing, El2N_performance_batch), 0
                ).to(device)

            El2N_performance_epoch.append(El2N_performance_before_debiasing)

    El2N_performance_all = torch.cat(
        [
            torch.unsqueeze(El2N_performance_epoch[i], dim=0)
            for i in range(len(El2N_performance_epoch))
        ]
    )
    El2N_performance_mean = torch.mean(
        torch.norm(El2N_performance_all, dim=2, p=2), dim=0
    ).to(device)

    return El2N_performance_mean
