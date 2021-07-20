from torch.optim import Adam
import pandas as pd
import numpy as np
import torch
from classifier import train_classifier
import wandb
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softmax = torch.nn.Softmax(dim=1).to(device)


def training_epoch(
    epoch, args, optimizer, device, tokenizer, model, train_data, train_data_gender_swap
):
    """
    Apply policy gradient approach for 1 epoch of the trianing data.
    args:
        epoch: the current epoch
        args: the arguments given by the user
        optimizer: the optimizer used to minize the loss
        device: the current device (cpu or gpu)
        tokenizer: the tokenizer used before giving the sentences to the classifier
        model: the pretrained classifier
        train_data: the training data
        train_data_gender_swap: the training data after swapping the genders (from male to female and vice versa)
    returns:
        loss (torch.tensor): the training epoch loss
    """
    logs = dict()
    model.train()
    compute_gradient = True
    epoch_bias, epoch_accuracy, epoch_reward, loss = epoch_loss(
        epoch,
        args,
        optimizer,
        device,
        tokenizer,
        model,
        train_data,
        train_data_gender_swap,
        compute_gradient,
    )

    #### Log everything
    logs["training_bias"] = -epoch_bias.cpu().numpy() / (
        len(train_data) / args.batch_size
    )
    logs["training_reward_mean"] = epoch_reward.cpu().numpy() / (
        len(train_data) / args.batch_size
    )
    logs["training_accuracy"] = epoch_accuracy.cpu().numpy() / (
        len(train_data) / args.batch_size
    )
    logs["epoch"] = epoch
    wandb.log(logs)

    return loss


def validation_epoch(
    epoch,
    args,
    optimizer,
    device,
    tokenizer,
    model,
    best_validation_reward,
    validation_data,
    validation_data_gender_swap,
):
    """
    Apply policy gradient approach for 1 epoch of the validation data.
    args:
        epoch: the current epoch
        args: the arguments given by the user
        optimizer: the optimizer used to minize the loss
        device: the current device (cpu or gpu)
        tokenizer: the tokenizer used before giving the sentences to the classifier
        model: the pretrained classifier
        best_validation_reward: the best validation reward that we use for model selection
        validation_data: the validation data
        validation_data_gender_swap: the validation data after swapping the genders (from male to female and vice versa)
    returns:
        loss (torch.tensor): the validation epoch loss
    """
    logs = dict()
    model.eval()
    compute_gradient = False

    with torch.no_grad():
        epoch_bias, epoch_accuracy, epoch_reward, loss = epoch_loss(
            epoch,
            args,
            optimizer,
            device,
            tokenizer,
            model,
            validation_data,
            validation_data_gender_swap,
            compute_gradient,
        )
        #### Log everything
        ## Todo: check this division
        logs["validation_bias"] = -epoch_bias.cpu().numpy() / (
            len(validation_data) / args.batch_size
        )
        logs["validation_reward_mean"] = epoch_reward.cpu().numpy() / (
            len(validation_data) / args.batch_size
        )
        logs["validation_accuracy"] = epoch_accuracy.cpu().numpy() / (
            len(validation_data) / args.batch_size
        )
        logs["epoch"] = epoch
        wandb.log(logs)

        # if the developmenet accuracy is better than the bext developement reward, we save the model weights.
        validation_reward_mean = epoch_reward.cpu().numpy() / (
            np.floor(len(validation_data) / args.batch_size)
        )
        if validation_reward_mean > best_validation_reward:
            best_validation_reward = validation_reward_mean
            torch.save(
                model.state_dict(),
                "./saved_models/" + args.classifier_model + "_debiased.pt",
            )

    return loss, best_validation_reward


def test_epoch(
    epoch,
    args,
    optimizer,
    device,
    tokenizer,
    model,
    test_data,
    test_data_gender_swap,
):
    """
    Apply policy gradient approach for 1 epoch of the test data.
    args:
        epoch: the current epoch
        args: the arguments given by the user
        optimizer: the optimizer used to minize the loss
        device: the current device (cpu or gpu)
        tokenizer: the tokenizer used before giving the sentences to the classifier
        model: the pretrained classifier
        test: the test data
        test: the test data after swapping the genders (from male to female and vice versa)
    returns:
        loss (torch.tensor): the test epoch loss
    """
    model.eval()
    compute_gradient = False

    with torch.no_grad():
        epoch_bias, epoch_accuracy, epoch_reward, loss = epoch_loss(
            epoch,
            args,
            optimizer,
            device,
            tokenizer,
            model,
            test_data,
            test_data_gender_swap,
            compute_gradient,
        )


    return loss, epoch_accuracy


def epoch_loss(
    epoch,
    args,
    optimizer,
    device,
    tokenizer,
    model,
    data,
    data_gender_swap,
    compute_gradient,
):
    """
    Apply policy gradient approach for 1 epoch of the validation data.
    args:
        epoch: the current epoch
        args: the arguments given by the user
        optimizer: the optimizer used to minize the loss
        device: the current device (cpu or gpu)
        tokenizer: the tokenizer used before giving the sentences to the classifier
        model: the pretrained classifier
        data: the training/validation data
        data_gender_swap: the training/validation data after swapping the genders (from male to female and vice versa)
        compute_gradient: a boolean that indicates whether or not we should compute the gradient of the loss
    returns:
        epoch_bias: the epoch average bias, which is the l2 norm of the difference between the logits of the model in 2 cases. The first case is when given the original sentence and the second is after swapping the gender in the sentences. The bias is averaged over the number of batches in the epoch.
        epoch_accuracy: the epoch average accuracy
        epoch_reward: the average epoch reward due to both accuracy and bias, which computed as (bias_reward + lambda * accuracy_reward)
        loss: the  epoch loss
    """
    epoch_bias = torch.tensor(0.0).to(device)
    epoch_accuracy = torch.tensor(0.0).to(device)
    epoch_reward = torch.tensor(0.0).to(device)
    epoch_loss = torch.tensor(0.0).to(device)

    input_column_name = data.columns[1]
    input_column_name_gender_swap = data_gender_swap.columns[1]
    label_column_name = data.columns[2]

    for i in range(int(np.ceil(len(data) / args.batch_size))):

        rewards_acc, rewards_bias, rewards_total, accuracy = [], [], [], []

        #### get a batch from the dataset
        df_batch_original_gender = list(
            data[input_column_name].iloc[
                i * args.batch_size : (i + 1) * args.batch_size
            ]
        )
        df_batch_original_gender_tokenized = tokenizer(
            df_batch_original_gender,
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )

        df_batch_gender_swap = list(
            data_gender_swap[input_column_name_gender_swap].iloc[
                i * args.batch_size : (i + 1) * args.batch_size
            ]
        )
        df_batch_gender_swap_tokenized = tokenizer(
            df_batch_gender_swap,
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )

        results_original_gender = model.forward(
            input_ids=torch.tensor(df_batch_original_gender_tokenized["input_ids"]).to(
                device
            ),
            attention_mask=torch.tensor(
                df_batch_original_gender_tokenized["attention_mask"]
            ).to(device),
            token_type_ids=torch.tensor(
                df_batch_original_gender_tokenized["token_type_ids"]
            ).to(device),
        )[0]
        results_gender_swap = model.forward(
            input_ids=torch.tensor(df_batch_gender_swap_tokenized["input_ids"]).to(
                device
            ),
            attention_mask=torch.tensor(
                df_batch_gender_swap_tokenized["attention_mask"]
            ).to(device),
            token_type_ids=torch.tensor(
                df_batch_gender_swap_tokenized["token_type_ids"]
            ).to(device),
        )[0]

        if(args.norm=="l1"):
          reward_bias = -torch.norm(
              results_original_gender - results_gender_swap, dim=1, p=1
          ).to(device)

        elif(args.norm=="l2"):
          reward_bias = -torch.norm(
              results_original_gender - results_gender_swap, dim=1, p=2
          ).to(device)

        reward_acc = (
            torch.argmax(results_original_gender, axis=1)
            == torch.tensor(
                data[label_column_name]
                .iloc[i * args.batch_size : (i + 1) * args.batch_size]
                .tolist()
            ).to(device)
        ).double()

        rewards_bias.append(torch.tensor(reward_bias))
        rewards_acc.append(torch.tensor(reward_acc))
        rewards_total.append(torch.tensor(reward_bias + args.lambda_PG * reward_acc))

        accuracy.append(
            torch.argmax(results_original_gender, axis=1)
            == torch.tensor(
                data[label_column_name]
                .iloc[i * args.batch_size : (i + 1) * args.batch_size]
                .tolist()
            ).to(device)
        )
        rewards = torch.cat(rewards_total)

        epoch_bias += torch.sum(torch.stack(rewards_bias)) / args.batch_size
        epoch_accuracy += torch.sum(torch.stack(accuracy)) / args.batch_size
        epoch_reward += torch.sum(rewards) / args.batch_size

        #### Run the policy gradient algorithm
        loss = -torch.sum(
            torch.log(torch.max(softmax(results_original_gender), axis=1)[0]) * rewards
        )
        if compute_gradient == True:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return epoch_bias, epoch_accuracy, epoch_reward, loss


def run_experiment(args):
    """
    Run the experiment by passing over the training and validation data to fine-tune the pretrained model
    args:
        args: the arguments given by the user
    returns:
        model: the model after updating its weights based on the policy gradient algorithm.
        tokenizer: the tokenizer used before giving the sentences to the classifier
    """
    wandb.init(
        name=str(args.dataset) + " using " + str(args.norm) + " distance",
        project="reducing gender bias in " + str(args.dataset) + " using policy gradient",
        config=args,
    )
    # Define pretrained tokenizer and mode
    model, tokenizer = train_classifier(args)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate_PG)

    train_data = pd.read_csv("./data/" + args.dataset + "_train_original_gender.csv")
    train_data_gender_swap = pd.read_csv(
        "./data/" + args.dataset + "_train_gender_swap.csv"
    )
    validation_data = pd.read_csv(
        "./data/" + args.dataset + "_valid_original_gender.csv"
    )
    validation_data_gender_swap = pd.read_csv(
        "./data/" + args.dataset + "_valid_gender_swap.csv"
    )
    
    test_data = pd.read_csv(
        "./data/" + args.dataset + "_valid_original_gender.csv"
    )
    test_data_gender_swap = pd.read_csv(
        "./data/" + args.dataset + "_valid_gender_swap.csv"
    )    

    best_validation_reward = torch.tensor(-float('inf')).to(device)
    for epoch in range(args.num_epochs_PG):
        training_loss = training_epoch(
            epoch,
            args,
            optimizer,
            device,
            tokenizer,
            model,
            train_data,
            train_data_gender_swap,
        )
        validation_loss, best_validation_reward = validation_epoch(
            epoch,
            args,
            optimizer,
            device,
            tokenizer,
            model,
            best_validation_reward,
            validation_data,
            validation_data_gender_swap,
        )

    # Load the model that has the best performance on the validation data
    model.load_state_dict(
        torch.load(
            "./saved_models/" + args.classifier_model + "_debiased.pt",
            map_location=device,
        )
    )
    
    # Compute the test accuracy and loss using the model with the highest validation accuracy
    test_loss, test_accuracy = test_epoch(
        epoch,
        args,
        optimizer,
        device,
        tokenizer,
        model,
        test_data,
        test_data_gender_swap,
    )        
    
    # Divide the test accuracy over all the batches by the number of batches that we have
    test_accuracy = test_accuracy / (
        len(test_data) / args.batch_size
    )    
    
    # Save the test accuracy
    output_file = "./output/test_accuracy.json"
    with open(output_file, "w+") as f:
        json.dump(str(test_accuracy), f, indent=2)        


    return model, tokenizer
