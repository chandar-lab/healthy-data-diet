from torch.optim import Adam
import numpy as np
import torch
from model.classifier import train_classifier
from model.data_loader import data_loader
import wandb
import json
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softmax = torch.nn.Softmax(dim=1).to(device)


def training_epoch(
    epoch,
    args,
    optimizer,
    device,
    tokenizer,
    model,
    train_dataset,
):
    """
    Compute the loss on the training data. 
    args:
        epoch: the current epoch
        args: the arguments given by the user
        optimizer: the optimizer used to minize the loss
        device: the current device (cpu or gpu)
        tokenizer: the tokenizer used before giving the sentences to the classifier
        train_dataset: the training data object
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
        train_dataset,
        compute_gradient,
    )

    #### Log everything
    # We divide the bias by this factor to be able to compare it for different lambdas
    logs["training_bias"] = epoch_bias.cpu().numpy()/(np.sqrt(args.lambda_data)**2 + np.sqrt(args.lambda_gender)**2)
    if args.approach == "policy_gradient":
        # We only log the reward if we are doing policy gradient
        logs["training_reward_mean"] = epoch_reward.cpu().numpy()
    logs["training_accuracy"] = epoch_accuracy.cpu().numpy()
    logs["epoch"] = epoch + 1
    if(args.use_wandb):
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
    val_dataset,
):
    """
    Compute the loss on the validation data. 
    args:
        epoch: the current epoch
        args: the arguments given by the user
        optimizer: the optimizer used to minize the loss
        device: the current device (cpu or gpu)
        tokenizer: the tokenizer used before giving the sentences to the classifier
        model: the pretrained classifier
        best_validation_reward: the best validation reward that we use for model
        selection
        val_dataset: the validation data object
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
            val_dataset,
            compute_gradient,
        )

        #### Log everything
        # We divide the bias by this factor so that we can compare the bias for different values of lambda.
        logs["validation_bias"] = (epoch_bias.cpu().numpy())/(np.sqrt(args.lambda_data)**2 + np.sqrt(args.lambda_gender)**2)
        if args.approach == "policy_gradient":
            # We only log the reward if we are doing policy gradient
            logs["validation_reward_mean"] = epoch_reward.cpu().numpy()
        logs["validation_accuracy"] = epoch_accuracy.cpu().numpy()
        # We add 1 because python starts with 0 instead of 1
        logs["epoch"] = epoch + 1
        if args.use_wandb:
            wandb.log(logs)

        # if the developmenet accuracy is better than the best developement
        # reward, we save the model weights.
        # We divide by the maximum bias to make sure both the accureacy and bias
        # are between 0 and 1. We add a small number in the denominator to
        # avoid dividing by 0
        if epoch_accuracy > best_validation_reward:
            best_validation_reward = epoch_accuracy
            torch.save(
                model.state_dict(),
                "./saved_models/"
                + args.classifier_model
                + "_"
                + args.method
                + "_"
                + args.approach
                + "_"
                + args.dataset
                +
                "_aux_loss_"
                +
                str(args.use_auxiliary_loss)                
                + "_"
                + args.norm
                + "_"
                +str(args.lambda_gender)
                + "_"
                + str(args.lambda_data)                   
                + "_debiased_best.pt",
            )

    return loss, best_validation_reward


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
    Compute the loss on the test data. 
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
            test_dataset,
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
    dataset,
    compute_gradient,
):
    """
    Compute the loss for 1 epoch.
    args:
        epoch: the current epoch
        args: the arguments given by the user
        optimizer: the optimizer used to minize the loss
        device: the current device (cpu or gpu)
        tokenizer: the tokenizer used before giving the sentences to the
        classifier
        model: the pretrained classifier
        data: the training/validation/test data objects
        compute_gradient: a boolean that indicates whether or not we should
        compute the gradient of the loss
    returns:
        epoch_bias: the epoch average bias, which is the l2 norm of the
        difference between the logits of the model due to 2 sources. The first
        source is gender bias, which we measure by flipping the gender and taking
        the norm of the difference in the logits before and after gender flipping.
        The second source is unintended correlationm which we measure by
        praphrasing the sentence and also taking the norm of the differenc
        between the logits before and after paraphrasing.
        epoch_accuracy: the epoch average accuracy
        epoch_reward: the average epoch reward due to both accuracy and bias,
        which computed as (bias_reward + lambda * accuracy_reward)
        loss: the  epoch loss
    """
    epoch_bias = torch.tensor(0.0).to(device)
    epoch_accuracy = torch.tensor(0.0).to(device)
    epoch_reward = torch.tensor(0.0).to(device)
    criterion = nn.CrossEntropyLoss()

    # Choosing the distance metric in the loss function
    if args.norm == "l1":
        norm_p_value = 1
    elif args.norm == "l2":
        norm_p_value = 2

    for i in range(int(np.ceil(len(dataset) / args.batch_size))):

        results_original_gender = model.forward(
            input_ids=torch.tensor(
                dataset.encodings["input_ids"][
                    i * args.batch_size : (i + 1) * args.batch_size
                ]
            ).to(device),
            attention_mask=torch.tensor(
                dataset.encodings["attention_mask"][
                    i * args.batch_size : (i + 1) * args.batch_size
                ]
            ).to(device),
            token_type_ids=torch.tensor(
                dataset.encodings["token_type_ids"][
                    i * args.batch_size : (i + 1) * args.batch_size
                ]
            ).to(device),
        )[0]
        
        if args.use_auxiliary_loss:
          results_gender_swap = model.forward(
              input_ids=torch.tensor(
                  dataset.encodings_gender_swap["input_ids"][
                      i * args.batch_size : (i + 1) * args.batch_size
                  ]
              ).to(device),
              attention_mask=torch.tensor(
                  dataset.encodings_gender_swap["attention_mask"][
                      i * args.batch_size : (i + 1) * args.batch_size
                  ]
              ).to(device),
              token_type_ids=torch.tensor(
                  dataset.encodings_gender_swap["token_type_ids"][
                      i * args.batch_size : (i + 1) * args.batch_size
                  ]
              ).to(device),
          )[0]
        # CLP is the counterfactual logit pairing baseline in https://arxiv.org/abs/1809.10610
        if args.use_auxiliary_loss or args.method=="CLP":  
            results_pharaphrasing = model.forward(
                input_ids=torch.tensor(
                    dataset.encodings_paraphrasing["input_ids"][
                        i * args.batch_size : (i + 1) * args.batch_size
                    ]
                ).to(device),
                attention_mask=torch.tensor(
                    dataset.encodings_paraphrasing["attention_mask"][
                        i * args.batch_size : (i + 1) * args.batch_size
                    ]
                ).to(device),
                token_type_ids=torch.tensor(
                    dataset.encodings_paraphrasing["token_type_ids"][
                        i * args.batch_size : (i + 1) * args.batch_size
                    ]
                ).to(device),
            )[0]

        if args.approach == "policy_gradient":
            rewards_acc, rewards_bias, rewards_total = [], [], []

            if len(dataset.encodings["input_ids"]) == len(
                dataset.encodings_gender_swap["input_ids"]
            ):
                # We can only compute the bias if the number of examples in data
                # and data_gender_swap is the same
                reward_bias_gender = -args.lambda_gender * torch.norm(
                    results_original_gender - results_gender_swap, dim=1, p=norm_p_value
                ).to(device)
                reward_bias_data = -args.lambda_data * torch.norm(
                    results_original_gender - results_pharaphrasing,
                    dim=1,
                    p=norm_p_value,
                ).to(device)
                reward_bias = reward_bias_gender + reward_bias_data

            else:
                # If the nummber of examples in data and data_gender_swap is
                # different, we set the bias to an arbitrary value of -1,
                # meaning that we cant compute the bias.
                reward_bias = torch.tensor(-1).to(device)

            rewards_bias.append(torch.tensor(reward_bias))

            reward_acc = (
                torch.argmax(results_original_gender, axis=1)
                == torch.tensor(
                    dataset.labels[i * args.batch_size : (i + 1) * args.batch_size]
                ).to(device)
            ).double()

            rewards_total.append(torch.tensor(reward_bias + reward_acc))

            rewards_acc.append(reward_acc)
            rewards = torch.cat(rewards_total)

            epoch_bias -= torch.sum(torch.stack(rewards_bias)) / args.batch_size
            epoch_accuracy += torch.sum(torch.stack(rewards_acc)) / args.batch_size
            epoch_reward += torch.sum(rewards) / args.batch_size

            #### Run the policy gradient algorithm
            loss = -torch.sum(
                torch.log(torch.max(softmax(results_original_gender), axis=1)[0])
                * rewards
            )

        elif args.approach == "supervised_learning":

            batch_accuracy, batch_bias = [], []

            accuracy = (
                torch.argmax(results_original_gender, axis=1)
                == torch.tensor(
                    dataset.labels[i * args.batch_size : (i + 1) * args.batch_size]
                ).to(device)
            ).double()

            batch_accuracy.append(accuracy)

            if args.use_auxiliary_loss or args.method == "CLP":
                # We create a mask that lets us apply the bias penalty only to the examples that have a groundtruth label of 0
                mask = (1 - torch.tensor(dataset.labels[i * args.batch_size : (i + 1) * args.batch_size])).to(device)
                if args.use_auxiliary_loss:
                  bias_gender = args.lambda_gender * torch.norm(
                      results_original_gender - results_gender_swap,
                      dim=1,
                      p=norm_p_value,
                  ).to(device)
                bias_data = args.lambda_data * torch.norm(
                    results_original_gender - results_pharaphrasing,
                    dim=1,
                    p=norm_p_value,
                ).to(device)
                if args.use_auxiliary_loss and args.method != "CLP":
                    # bias = bias_data + torch.mul(bias_gender,mask)
                    bias = bias_data + bias_gender
                elif args.use_auxiliary_loss==False and args.method == "CLP":
                    bias = torch.mul(bias_data,mask)
                    # if the method used is CLP, then we only need the bias due to counterfactual logit difference
                    # https://dl.acm.org/doi/pdf/10.1145/3306618.3317950
                    
                    # this mask is true when the sentence is not toxic and vice versa, as explained in the paper.
                    # We only consider the bias term when the sentence is not toxic
                else:
                    # If we are using CLP and the auxiliary loss
                    # bias = torch.mul(bias_data,mask) + torch.mul(bias_gender,mask)
                    bias = torch.mul(bias_data,mask) + bias_gender
                    
                batch_bias.append(torch.tensor(bias))
                epoch_bias += torch.sum(torch.stack(batch_bias)) / args.batch_size

            epoch_accuracy += torch.sum(torch.stack(batch_accuracy)) / args.batch_size

            output_dim = results_original_gender.shape[1]

            if args.method in ["baseline_data_augmentation","baseline_data_substitution"] and args.use_auxiliary_loss==False:
                # This baseline method fine-tunes based only on cross entropy,
                # without the additional term for the loss due to bias term
                # (unlike ours).
                targets = torch.tensor(
                    dataset.labels[i * args.batch_size : (i + 1) * args.batch_size]
                ).to(device)
                loss = criterion(
                    results_original_gender.contiguous().view(-1, output_dim),
                    targets.contiguous().view(-1),
                )
            elif args.use_auxiliary_loss or args.method == "CLP":
                targets = torch.tensor(
                    dataset.labels[i * args.batch_size : (i + 1) * args.batch_size]
                ).to(device)
                loss = (
                    criterion(
                        results_original_gender.contiguous().view(-1, output_dim),
                        targets.contiguous().view(-1),
                    )
                    + torch.mean(torch.Tensor.float(bias)).item()
                )
        if compute_gradient:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # This correction factor is to compute the mean over the whole dataset
    epoch_bias /= len(dataset) / args.batch_size
    epoch_accuracy /= len(dataset) / args.batch_size
    epoch_reward /= len(dataset) / args.batch_size
    loss /= len(dataset) / args.batch_size

    return epoch_bias, epoch_accuracy, epoch_reward, loss


def run_experiment(args, run):
    """
    Run the experiment by passing over the training and validation data to
    fine-tune the pretrained model.
    Compute the test accuracy on the whole test set.
    Save the text accuracy in json files.
    args:
        args: the arguments given by the user
        run : the index of the current run, that goes from 0 to the number of runs defined by the user
    returns:
        model: the model after updating its weights based on the policy
        gradient algorithm.
        tokenizer: the tokenizer used before giving the sentences to the classifier
    """
    if(args.use_wandb):
        wandb.init(
            name=str(args.dataset) + "_run_" + str(run),
            project="reducing gender bias",
            config=args,
        )
    # Define pretrained tokenizer and mode
    model, tokenizer = train_classifier(args)
    
    # save the best biased model
    torch.save(
        model.state_dict(),
        "./saved_models/"
        + args.classifier_model
        + "_"
        + args.method
        + "_"
        + args.approach
        + "_"
        + args.dataset
        +
        "_aux_loss_"
        +
        str(args.use_auxiliary_loss)                
        + "_"
        + args.norm
        + "_"
        +str(args.lambda_gender)
        + "_"
        + str(args.lambda_data)                   
        + "_biased_best.pt",
    )
            
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # Load the dataset
    train_dataset, val_dataset, test_dataset = data_loader(args, apply_data_augmentation = (args.method == "baseline_data_augmentation"), apply_data_substitution = (args.method == "baseline_data_substitution"))

    best_validation_reward = torch.tensor(-float("inf")).to(device)
    for epoch in range(args.num_epochs):
        training_loss = training_epoch(
            epoch,
            args,
            optimizer,
            device,
            tokenizer,
            model,
            train_dataset,
        )
        validation_loss, best_validation_reward = validation_epoch(
            epoch,
            args,
            optimizer,
            device,
            tokenizer,
            model,
            best_validation_reward,
            val_dataset,
        )
        if epoch < args.num_saved_debiased_models and args.analyze_results:
            torch.save(
                model.state_dict(),
                "./saved_models/"
                + args.classifier_model
                + "_"
                + args.method
                + "_"
                + args.approach
                + "_"
                + args.dataset
                +
                "_aux_loss_"
                +
                str(args.use_auxiliary_loss)                
                + "_"
                + args.norm
                + "_"
                +str(args.lambda_gender)
                + "_"
                + str(args.lambda_data)          
                + "_debiased_epoch_"
                + str(epoch)
                + ".pt",
            )

    # Load the model that has the best performance on the validation data
    model.load_state_dict(
        torch.load(
            "./saved_models/"
            + args.classifier_model
            + "_"
            + args.method
            + "_"
            + args.approach
            + "_"
            + args.dataset
            +
            "_aux_loss_"
            +
            str(args.use_auxiliary_loss)            
            + "_"
            + args.norm
            + "_"
            + str(args.lambda_gender)
            + "_"
            + str(args.lambda_data)             
            + "_debiased_best.pt",
            map_location=device,
        )
    )

    # Compute the test accuracy and loss using the model with the highest
    # validation accuracy
    test_loss, test_accuracy = test_epoch(
        epoch,
        args,
        optimizer,
        device,
        tokenizer,
        model,
        test_dataset,
    )

    # Save the test accuracy
    output_file = (
        "./output/accuracy/test_accuracy_"
        + args.method
        + "_"
        + args.approach
        + "_"
        + args.dataset
        +
        "_aux_loss_"
        +
        str(args.use_auxiliary_loss)        
        + "_"
        + args.norm
        + "_"
        +str(args.lambda_gender)
        + "_"
        + str(args.lambda_data)         
        + ".json"
    )
    with open(output_file, "w+") as f:
        json.dump(test_accuracy.tolist(), f, indent=2)


    return model, tokenizer
