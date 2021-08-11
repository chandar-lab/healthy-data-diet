from torch.optim import Adam
import numpy as np
import torch
from models.classifier import train_classifier
from models.data_loader import data_loader
import wandb
import json
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softmax = torch.nn.Softmax(dim=1).to(device)


def training_epoch(
    epoch, args, optimizer, device, tokenizer, model, train_dataset,
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
        train_dataset: the training data object
    returns:
        loss (torch.tensor): the training epoch loss
        model: the pretrained classifier
    """
    logs = dict()
    model.train()
    compute_gradient = True
    epoch_bias, epoch_accuracy, epoch_reward, loss, model = epoch_loss(
        epoch,
        args,
        optimizer,
        device,
        tokenizer,
        model,
        train_dataset,
        compute_gradient,
    )

    #### Log everything
    logs["training_bias"] = epoch_bias.cpu().numpy() / (
        train_dataset.__len__() / args.batch_size
    )
    if(args.approach=="policy_gradient"):
        # We only log the reward if we are doing policy gradient
        logs["training_reward_mean"] = epoch_reward.cpu().numpy() / (
            train_dataset.__len__() / args.batch_size
        )
    logs["training_accuracy"] = epoch_accuracy.cpu().numpy() / (
        train_dataset.__len__() / args.batch_size
    )
    logs["epoch"] = epoch +1
    wandb.log(logs)

    return loss, model


def validation_epoch(
    epoch,
    args,
    optimizer,
    device,
    tokenizer,
    model,
    best_validation_reward,
    val_dataset,
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
        val_dataset: the validation data object
    returns:
        loss (torch.tensor): the validation epoch loss
        model: the pretrained classifier
    """
    logs = dict()
    model.eval()
    compute_gradient = False

    with torch.no_grad():
        epoch_bias, epoch_accuracy, epoch_reward, loss, model = epoch_loss(
            epoch,
            args,
            optimizer,
            device,
            tokenizer,
            model,
            val_dataset,
            compute_gradient,
        )

        # We divide by this factor to make sure that the values averaged average over the whole epoch.
        epoch_accuracy /= (np.floor(val_dataset.__len__() / args.batch_size)) 
        epoch_bias /= (np.floor(val_dataset.__len__() / args.batch_size)) 
        epoch_reward /= (np.floor(val_dataset.__len__() / args.batch_size)) 
        #### Log everything
        logs["validation_bias"] = epoch_bias.cpu().numpy()
        if(args.approach=="policy_gradient"):
            # We only log the reward if we are doing policy gradient
            logs["validation_reward_mean"] = epoch_reward.cpu().numpy() 
        logs["validation_accuracy"] = epoch_accuracy.cpu().numpy() 
        # We add 1 because python starts with 0 instead of 1
        logs["epoch"] = epoch + 1
        wandb.log(logs)
 
        # if the developmenet accuracy is better than the best developement reward, we save the model weights. 
        if (epoch_accuracy - epoch_bias)> best_validation_reward:
            print(epoch+1)
            print(epoch_accuracy, epoch_bias)
            best_validation_reward = epoch_accuracy - epoch_bias
            torch.save(
                model.state_dict(),
                "./saved_models/" + args.classifier_model + "_debiased_best.pt",
            )

    return loss, best_validation_reward, model


def test_epoch(
    epoch,
    args,
    optimizer,
    device,
    tokenizer,
    model,
    test_dataset,
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
        test_dataset: the test data object
    returns:
        loss (torch.tensor): the test epoch loss
        model: the pretrained classifier
    """
    model.eval()
    compute_gradient = False

    with torch.no_grad():
        epoch_bias, epoch_accuracy, epoch_reward, loss, model = epoch_loss(
            epoch,
            args,
            optimizer,
            device,
            tokenizer,
            model,
            test_dataset,
            compute_gradient,
        )

    return loss, epoch_accuracy, model


def epoch_loss(
    epoch,
    args,
    optimizer,
    device,
    tokenizer,
    model,
    dataset,
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
        data: the training/validation/test data objects
        compute_gradient: a boolean that indicates whether or not we should compute the gradient of the loss
    returns:
        epoch_bias: the epoch average bias, which is the l2 norm of the difference between the logits of the model due to 2 sources. The first source is gender bias, which we measure by flipping the gender and taking the norm of the difference in the logits before and after gender flipping. The second source is unintended correlationm which we measure by praphrasing the sentence and also taking the norm of the differenc between the logits before and after paraphrasing.
        epoch_accuracy: the epoch average accuracy
        epoch_reward: the average epoch reward due to both accuracy and bias, which computed as (bias_reward + lambda * accuracy_reward)
        loss: the  epoch loss
        model: the pretrained classifier
    """
    epoch_bias = torch.tensor(0.0).to(device)
    epoch_accuracy = torch.tensor(0.0).to(device)
    epoch_reward = torch.tensor(0.0).to(device)
    criterion = nn.CrossEntropyLoss()
    

    for i in range(int(np.ceil(dataset.__len__() / args.batch_size))):

        results_original_gender = model.forward(
            input_ids=torch.tensor(dataset.encodings["input_ids"][i*args.batch_size : (i + 1) * args.batch_size]).to(
                device
            ),    
            attention_mask=torch.tensor(
                dataset.encodings["attention_mask"][i*args.batch_size : (i + 1) * args.batch_size]
            ).to(device),
            token_type_ids=torch.tensor(
                dataset.encodings["token_type_ids"][i*args.batch_size : (i + 1) * args.batch_size]
            ).to(device)
        )[0]
        results_gender_swap = model.forward(
            input_ids=torch.tensor(dataset.encodings_gender_swap["input_ids"][i*args.batch_size : (i + 1) * args.batch_size]).to(
                device
            ),    
            attention_mask=torch.tensor(
                dataset.encodings_gender_swap["attention_mask"][i*args.batch_size : (i + 1) * args.batch_size]
            ).to(device),
            token_type_ids=torch.tensor(
                dataset.encodings_gender_swap["token_type_ids"][i*args.batch_size : (i + 1) * args.batch_size]
            ).to(device)
        )[0]
        results_pharaphrasing = model.forward(
            input_ids=torch.tensor(dataset.encodings_paraphrasing["input_ids"][i*args.batch_size : (i + 1) * args.batch_size]).to(
                device
            ),    
            attention_mask=torch.tensor(
                dataset.encodings_paraphrasing["attention_mask"][i*args.batch_size : (i + 1) * args.batch_size]
            ).to(device),
            token_type_ids=torch.tensor(
                dataset.encodings_paraphrasing["token_type_ids"][i*args.batch_size : (i + 1) * args.batch_size]
            ).to(device)
        )[0]        
        
        if(args.approach == "policy_gradient"):
            rewards_acc, rewards_bias, rewards_total = [], [], []
    
    
            if len(dataset.encodings["input_ids"]) == len(dataset.encodings_gender_swap["input_ids"]):
                # We can only compute the bias if the number of examples in data and data_gender_swap is the same
                if args.norm == "l1":
                    reward_bias = - args.lambda_gender * torch.norm(
                        results_original_gender - results_gender_swap, dim=1, p=1
                    ).to(device) - args.lambda_data * torch.norm(
                        results_original_gender - results_pharaphrasing, dim=1, p=1
                    ).to(device)
    
                elif args.norm == "l2":
                    reward_bias = - args.lambda_gender * torch.norm(
                        results_original_gender - results_gender_swap, dim=1, p=2
                    ).to(device) -args.lambda_data * torch.norm(
                        results_original_gender - results_pharaphrasing, dim=1, p=2
                    ).to(device)
    
            else:
                # If the nummber of examples in data and data_gender_swap is different, we set the bias to an arbitrary value of -1, meaning that we cant compute the bias.
                reward_bias = torch.tensor(-1).to(device)
    
            rewards_bias.append(torch.tensor(reward_bias))
    
            reward_acc = (
                torch.argmax(results_original_gender, axis=1)
                == torch.tensor(dataset.labels[i * args.batch_size : (i + 1) * args.batch_size]).to(device)
            ).double()
    
            rewards_total.append(torch.tensor(reward_bias + reward_acc))
    
            rewards_acc.append(reward_acc)
            rewards = torch.cat(rewards_total)
    
            epoch_bias -= torch.sum(torch.stack(rewards_bias)) / args.batch_size
            epoch_accuracy += torch.sum(torch.stack(rewards_acc)) / args.batch_size
            epoch_reward += torch.sum(rewards) / args.batch_size
    
            #### Run the policy gradient algorithm
            loss = -torch.sum(
                torch.log(torch.max(softmax(results_original_gender), axis=1)[0]) * rewards
            )
            
        elif(args.approach == "supervised_learning"):
            
            batch_accuracy, batch_bias = [], []

            accuracy = (
                torch.argmax(results_original_gender, axis=1)
                == torch.tensor(dataset.labels[i * args.batch_size : (i + 1) * args.batch_size]).to(device)
            ).double()
        
            batch_accuracy.append(accuracy)

            
            if len(dataset.encodings["input_ids"]) == len(dataset.encodings_gender_swap["input_ids"]):
                # We can only compute the bias if the number of examples in data and data_gender_swap is the same
                if args.norm == "l1":
                    bias = torch.norm(
                        results_original_gender - results_gender_swap, dim=1, p=1
                    ).to(device)
    
                elif args.norm == "l2":
                    bias = torch.norm(
                        results_original_gender - results_gender_swap, dim=1, p=2
                    ).to(device)
    
            else:
                # If the nummber of examples in data and data_gender_swap is different, we set the bias to an arbitrary value of -1, meaning that we cant compute the bias.
                bias = torch.tensor(-1).to(device)            
        
            batch_bias.append(torch.tensor(bias))
            
            epoch_bias += torch.sum(torch.stack(batch_bias)) / args.batch_size
            epoch_accuracy += torch.sum(torch.stack(batch_accuracy)) / args.batch_size
            
            output_dim = results_original_gender.shape[1]
            trg = torch.tensor(dataset.labels[i * args.batch_size : (i + 1) * args.batch_size]).to(device)
            loss = criterion(results_original_gender.contiguous().view(-1, output_dim), trg.contiguous().view(-1)) + torch.mean(bias).item()             
        
        
        if compute_gradient == True:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return epoch_bias, epoch_accuracy, epoch_reward, loss, model


def run_experiment(args):
    """
    Run the experiment by passing over the training and validation data to fine-tune the pretrained model.
    Compute the test accuracy on the whole test set.
    Compute the test accuracy on the majority and majority groups of the test data. The majority groups refers to the set of examples where relying on the unintended correlation helps, whereas the minority group represens the set of exmaple where relying on unintended correlation hurts the performance.
    Save the text accuracy in json files.
    args:
        args: the arguments given by the user
    returns:
        model: the model after updating its weights based on the policy gradient algorithm.
        tokenizer: the tokenizer used before giving the sentences to the classifier
    """
    wandb.init(
        name=str(args.dataset) + " using " + str(args.norm) + " distance",
        project="reducing gender bias in "
        + str(args.dataset)
        + " using policy gradient",
        config=args,
    )
    # Define pretrained tokenizer and mode
    model, tokenizer = train_classifier(args)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate_PG)

    
    # Load the dataset
    train_dataset, val_dataset, test_dataset = data_loader(args)

    best_validation_reward = torch.tensor(-float("inf")).to(device)
    for epoch in range(args.num_epochs_PG):
        training_loss, model = training_epoch(
            epoch,
            args,
            optimizer,
            device,
            tokenizer,
            model,
            train_dataset,
        )
        validation_loss, best_validation_reward, model = validation_epoch(
            epoch,
            args,
            optimizer,
            device,
            tokenizer,
            model,
            best_validation_reward,
            val_dataset,
        )
        if epoch < args.num_saved_debiased_models:
            torch.save(
                model.state_dict(),
                "./saved_models/"
                + args.classifier_model
                + "_debiased_epoch_"
                + str(epoch)
                + ".pt",
            )

    # Load the model that has the best performance on the validation data
    model.load_state_dict(
        torch.load(
            "./saved_models/" + args.classifier_model + "_debiased_best.pt",
            map_location=device,
        )
    )

    # Compute the test accuracy and loss using the model with the highest validation accuracy
    test_loss, test_accuracy, model = test_epoch(
        epoch,
        args,
        optimizer,
        device,
        tokenizer,
        model,
        test_dataset,
    )

    # Divide the test accuracy over all the batches by the number of batches that we have
    test_accuracy = test_accuracy / (test_dataset.__len__() / args.batch_size)

    # Save the test accuracy
    output_file = "./output/test_accuracy.json"
    with open(output_file, "w+") as f:
        json.dump(str(test_accuracy), f, indent=2)

    if args.compute_majority_and_minority_accuracy:
        # Compute the test accuracy on the majority and majority groups of the test data.
        # The majority groups refers to the set of examples where relying on the unintended correlation improves the performance, whereas the minority group represents the set of exmaples where relying on unintended correlation hurts the performance.
        # Load the dataset
        _, _, test_dataset_majority = data_loader(args,test_subset="majority")
        _, _, test_dataset_minority = data_loader(args,test_subset="minority")
        _, test_accuracy_majority, model = test_epoch(
            epoch,
            args,
            optimizer,
            device,
            tokenizer,
            model,
            test_dataset_majority,
        )

        # Divide the test_accuracy_majority of all the batches by the number of batches that we have
        test_accuracy_majority = test_accuracy_majority / (
            test_dataset_majority.__len__() / args.batch_size
        )

        # Save the test accuracy on the majority group in the test data
        output_file = "./output/test_accuracy_majority.json"
        with open(output_file, "w+") as f:
            json.dump(str(test_accuracy_majority), f, indent=2)

        _, test_accuracy_minority, model = test_epoch(
            epoch,
            args,
            optimizer,
            device,
            tokenizer,
            model,
            test_dataset_minority,
        )

        # Divide the test_accuracy_minority of all the batches by the number of batches that we have
        test_accuracy_minority = test_accuracy_minority / (
            test_dataset_minority.__len__() / args.batch_size
        )

        # Save the test accuracy on the minority group of the test data
        output_file = "./output/test_accuracy_minority.json"
        with open(output_file, "w+") as f:
            json.dump(str(test_accuracy_minority), f, indent=2)

    return model, tokenizer
