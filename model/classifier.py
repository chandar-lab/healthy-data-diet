from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
from model.data_loader import data_loader
from sklearn.metrics import roc_auc_score
from pathlib import Path
import torch
import json
import numpy as np
import torch.nn.functional as F
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assess_performance_and_bias(model_after_bias_reduction, args, run):
    """
    Measure the performance before and after applying our de-biasing algorithm.
    This is done by computing the metrics on the validation, test and IPTTS datasets.
    The IPTTS dataset is a synthetic dataset, developed to measure the bias in the model.
    The metrics that we compute are the following:
    1) AUC
    2) Accuracy
    3) FNED
    4) FPED
    5) TNED
    6) TPED
    7) Demographic parity
    8) Equality of odds    
    args:
        model_after_bias_reduction: the model after updating its weights due to bias reduction
        args: the arguments given by the user
        run : the index of the current run, that goes from 0 to the number of runs defined by the user
    returns:
        the function doesnt return anything, since all the metrics are saved in json files.
    """
    all_metrics = []
    # We need to load the datasets on which we measure the metrics.
    train_dataset, val_dataset, test_dataset, IPTTS_dataset = data_loader(args, IPTTS=True, apply_data_augmentation = (args.method == "baseline_data_augmentation"), apply_data_substitution = (args.method == "baseline_data_substitution"))


    if args.method == "baseline_data_augmentation":
        # In data augmentation, the model before bias reduction had half the number of examples
        checkpoint_steps = (
            int((train_dataset.__len__()/2) / args.batch_size_pretraining)
        )
    else:
        checkpoint_steps = (
            int(train_dataset.__len__() / args.batch_size_pretraining)
        )         
    

    model_path = "./saved_models/checkpoint-" + str(checkpoint_steps)
    model_before_bias_reduction = BertForSequenceClassification.from_pretrained(
        model_path, num_labels=len(set(val_dataset.labels)), output_attentions=args.analyze_attention
    ).to(device)
    
    # Load the model that has the best performance on the validation data
    model_before_bias_reduction.load_state_dict(
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
            + "_biased_best.pt",
            map_location=device,
        )
    )        

    for split_name, split in zip(
        ["test", "validation", "IPTTS"], [test_dataset, val_dataset, IPTTS_dataset]
    ):

        # We compute the metrics that we need by calling this function
        AUC, accuracy, FPED, FNED, TPED, TNED, demographic_parity, equality_of_odds = compute_metrics(
            split,
            split_name,
            args.dataset,
            model_before_bias_reduction,
            model_after_bias_reduction,
            args.batch_size,
            args,
        )

        if(split_name == "IPTTS"):
            for metric in [FPED, FNED, TPED, TNED, demographic_parity, equality_of_odds]:             
                all_metrics.append(metric)
        else:            
            for metric in [AUC, accuracy]:
                all_metrics.append(metric)

    # We create a directory for saving the metrics, if it doesnt exist
    file_directory = (
        "./output/"
        + args.method
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
    )           
    
    Path(file_directory).mkdir(parents=True, exist_ok=True)        
    
    # We now save the metrics in a json file
    output_file = (file_directory + "/metrics.json")
    with open(output_file, "w+") as f:
        json.dump(all_metrics, f, indent=2)
                


def compute_metrics(
    split_dataset,
    split_name,
    dataset_name,
    model_before_bias_reduction,
    model_after_bias_reduction,
    batch_size,
    args,
):
    """
    Compute the performance and bias metrics before and after applying our
    de-biasing algorithm on a specific split from the dataset.
    args:
        split_dataset: the dataset split object on which the metrics are measured.
        split_name: the name of the split on which the metrics are computed.
        dataset_name: the nam of the dataset used
        model_before_bias_reduction: the model before updating its weights for bias reduction
        model_after_bias_reduction: the model after updating its weights for bias reduction
        batch_size: the size of the batch used
    returns:
        the function returns the folowing metrics: accuracy, AUC, FNED, and FPED.
    """
    with torch.no_grad():

        accuracy = {}
        AUC = {}
        FPR = {}
        FNR = {}
        TPR = {}
        TNR = {}        
        FPED = {}
        FNED = {}
        TPED = {}
        TNED = {}
        demographic_parity = {}
        equal_opportunity_y_equal_0 = {}
        equal_opportunity_y_equal_1 = {}
        equality_of_odds = {}
        
        num_labels = len(set(split_dataset.labels))
        y_pred_after_bias_reduction = torch.ones([0, num_labels]).to(device)
        y_pred_before_bias_reduction = torch.ones([0, num_labels]).to(device)


        for i in range(int(np.ceil(len(split_dataset) / batch_size))):

            results_after_bias_reduction = model_after_bias_reduction.forward(
                input_ids=torch.tensor(
                    split_dataset.encodings["input_ids"][
                        i * batch_size : (i + 1) * batch_size
                    ]
                ).to(device),
                attention_mask=torch.tensor(
                    split_dataset.encodings["attention_mask"][
                        i * batch_size : (i + 1) * batch_size
                    ]
                ).to(device),
                token_type_ids=torch.tensor(
                    split_dataset.encodings["token_type_ids"][
                        i * batch_size : (i + 1) * batch_size
                    ]
                ).to(device),
            )[0]

            results_before_bias_reduction = model_before_bias_reduction.forward(
                input_ids=torch.tensor(
                    split_dataset.encodings["input_ids"][
                        i * batch_size : (i + 1) * batch_size
                    ]
                ).to(device),
                attention_mask=torch.tensor(
                    split_dataset.encodings["attention_mask"][
                        i * batch_size : (i + 1) * batch_size
                    ]
                ).to(device),
                token_type_ids=torch.tensor(
                    split_dataset.encodings["token_type_ids"][
                        i * batch_size : (i + 1) * batch_size
                    ]
                ).to(device),
            )[0]

            # Get the predictions of the new batch
            batch_original_gender = results_after_bias_reduction
            # Add them to the total predictions
            y_pred_after_bias_reduction = torch.cat(
                (y_pred_after_bias_reduction, batch_original_gender), 0
            )

            # Get the predictions of the new batch
            batch_original_gender = results_before_bias_reduction
            # Add them to the total predictions
            y_pred_before_bias_reduction = torch.cat(
                (y_pred_before_bias_reduction, batch_original_gender), 0
            )
        # ===================================================#
        # Here we calculate the accuracy
        accuracy[split_name + " accuracy before bias reduction"] = (
            torch.sum(
                torch.argmax(y_pred_before_bias_reduction, axis=1).to(device)
                == torch.tensor(split_dataset.labels).to(device)
            )
            / len(split_dataset.labels)
        ).tolist()
        accuracy[split_name + " accuracy after bias reduction"] = (
            torch.sum(
                torch.argmax(y_pred_after_bias_reduction, axis=1).to(device)
                == torch.tensor(split_dataset.labels).to(device)
            )
            / len(split_dataset.labels)
        ).tolist()

        # ===================================================#
        # Here we calculate the AUC score
        AUC[split_name + " AUC before bias reduction"] = roc_auc_score(
            np.array(split_dataset.labels),
            np.array(F.softmax(y_pred_before_bias_reduction, 1).cpu())[:, 1],
        )
        AUC[split_name + " AUC after bias reduction"] = roc_auc_score(
            np.array(split_dataset.labels),
            np.array(F.softmax(y_pred_after_bias_reduction, 1).cpu())[:, 1],
        )

        # ===================================================#
        # Here we calculate the FNR

        FPR["before_bias_reduction"] = (
            torch.sum(
                torch.logical_and(
                    torch.argmax(y_pred_before_bias_reduction, axis=1).to(device) == 1,
                    torch.tensor(split_dataset.labels).to(device) == 0,
                )
            )
            / torch.sum(torch.tensor(split_dataset.labels).to(device) == 0)
        ).tolist()
        FPR["after_bias_reduction"] = (
            torch.sum(
                torch.logical_and(
                    torch.argmax(y_pred_after_bias_reduction, axis=1).to(device) == 1,
                    torch.tensor(split_dataset.labels).to(device) == 0,
                )
            )
            / torch.sum(torch.tensor(split_dataset.labels).to(device) == 0)
        ).tolist()

        # ===================================================#
        # Here we calculate the FPR

        FNR["before_bias_reduction"] = (
            torch.sum(
                torch.logical_and(
                    torch.argmax(y_pred_before_bias_reduction, axis=1).to(device) == 0,
                    torch.tensor(split_dataset.labels).to(device) == 1,
                )
            )
            / torch.sum(torch.tensor(split_dataset.labels).to(device) == 1)
        ).tolist()
        FNR["after_bias_reduction"] = (
            torch.sum(
                torch.logical_and(
                    torch.argmax(y_pred_after_bias_reduction, axis=1).to(device) == 0,
                    torch.tensor(split_dataset.labels).to(device) == 1,
                )
            )
            / torch.sum(torch.tensor(split_dataset.labels).to(device) == 1)
        ).tolist()
        
        # ===================================================#
        # Here we calculate the TNR

        TPR["before_bias_reduction"] = (
            torch.sum(
                torch.logical_and(
                    torch.argmax(y_pred_before_bias_reduction, axis=1).to(device) == 1,
                    torch.tensor(split_dataset.labels).to(device) == 1,
                )
            )
            / torch.sum(torch.tensor(split_dataset.labels).to(device) == 1)
        ).tolist()
        TPR["after_bias_reduction"] = (
            torch.sum(
                torch.logical_and(
                    torch.argmax(y_pred_after_bias_reduction, axis=1).to(device) == 1,
                    torch.tensor(split_dataset.labels).to(device) == 1,
                )
            )
            / torch.sum(torch.tensor(split_dataset.labels).to(device) == 1)
        ).tolist()

        # ===================================================#
        # Here we calculate the TPR

        TNR["before_bias_reduction"] = (
            torch.sum(
                torch.logical_and(
                    torch.argmax(y_pred_before_bias_reduction, axis=1).to(device) == 0,
                    torch.tensor(split_dataset.labels).to(device) == 0,
                )
            )
            / torch.sum(torch.tensor(split_dataset.labels).to(device) == 0)
        ).tolist()
        TNR["after_bias_reduction"] = (
            torch.sum(
                torch.logical_and(
                    torch.argmax(y_pred_after_bias_reduction, axis=1).to(device) == 0,
                    torch.tensor(split_dataset.labels).to(device) == 0,
                )
            )
            / torch.sum(torch.tensor(split_dataset.labels).to(device) == 0)
        ).tolist()

        # ===================================================#         
        # Here we calculate the FNED and FPED metrics, as described in https://arxiv.org/pdf/2004.14088.pdf
        if split_name == "IPTTS":          
            if dataset_name in ["Twitter_sexism_dataset", "Twitter_toxicity_dataset"]:
                data_IPTTS = pd.read_csv("./data/" + "madlib.csv")
                idxs_male = [
                    i
                    for i in range(len(data_IPTTS["gender"].values))
                    if data_IPTTS["gender"].values[i] == "male"
                ]
                idxs_female = [
                    i
                    for i in range(len(data_IPTTS["gender"].values))
                    if data_IPTTS["gender"].values[i] == "female"
                ]
            elif (dataset_name == "Wikipedia_toxicity_dataset" or dataset_name == "Jigsaw_toxicity_dataset"):
                data_IPTTS = pd.read_csv("./data/" + "bias_madlibs_77k.csv")
                idxs_male = list(data_IPTTS[data_IPTTS["text"].str.contains(" male")].index.values)
                idxs_female = list(data_IPTTS[data_IPTTS["text"].str.contains(" female")].index.values)  
                
            # Saving the model's predcitions before and after debiasing, on the 
            # IPTTs dataset.
            data_IPTTS["predcition before debiasing"] = list(
                F.softmax(y_pred_before_bias_reduction, 1)[:, 1].cpu().detach().numpy()
            )
            data_IPTTS["prediction after debiasing"] = list(
                F.softmax(y_pred_after_bias_reduction, 1)[:, 1].cpu().detach().numpy()
            )

            data_IPTTS.to_csv(
                "./output/analysis/bias_analysis"
                + "_"
                + args.method
                + "_"
                + args.approach
                + "_"
                + dataset_name
                +
                "_aux_loss_"
                +
                str(args.use_auxiliary_loss)        
                + "_"
                + args.norm        
                + ".csv",
                index=False,
            )          
            
            # We just initialize the FNED and FPED to zeros, before we compute them.
            FPED["FPED before bias reduction"], FPED["FPED after bias reduction"] = 0, 0
            FNED["FNED before bias reduction"], FNED["FNED after bias reduction"] = 0, 0
            TPED["TPED before bias reduction"], TPED["TPED after bias reduction"] = 0, 0
            TNED["TNED before bias reduction"], TNED["TNED after bias reduction"] = 0, 0   
            equal_opportunity_y_equal_0["before_bias_reduction"] = torch.tensor(0).to(device)
            equal_opportunity_y_equal_0["after_bias_reduction"] = torch.tensor(0).to(device)
            equal_opportunity_y_equal_1["before_bias_reduction"] = torch.tensor(0).to(device)
            equal_opportunity_y_equal_1["after_bias_reduction"] = torch.tensor(0).to(device)
            num_positive_examples = torch.sum(torch.tensor(split_dataset.labels).to(device) == 1)
            num_negative_examples = torch.sum(torch.tensor(split_dataset.labels).to(device) == 0)            
            for idx in [idxs_male, idxs_female]:
                num_positive_examples = torch.sum(torch.tensor(split_dataset.labels).to(device)[idx] == 1)
                num_negative_examples = torch.sum(torch.tensor(split_dataset.labels).to(device)[idx] == 0)                
                pred_before_bias_reduction = torch.argmax(y_pred_before_bias_reduction,axis = 1).to(device)[idx]
                pred_after_bias_reduction = torch.argmax(y_pred_after_bias_reduction,axis = 1).to(device)[idx]
                ground_truth = torch.tensor(split_dataset.labels).to(device)[idx]

                FPED["FPED before bias reduction"] += torch.abs(FPR["before_bias_reduction"]-(torch.sum(torch.logical_and(pred_before_bias_reduction == 1, ground_truth == 0))/num_negative_examples)).tolist()
                FPED["FPED after bias reduction"] += torch.abs(FPR["after_bias_reduction"]-(torch.sum(torch.logical_and(pred_after_bias_reduction == 1, ground_truth == 0))/num_negative_examples)).tolist()
  
                FNED["FNED before bias reduction"] += torch.abs(FNR["before_bias_reduction"]-(torch.sum(torch.logical_and(pred_before_bias_reduction == 0, ground_truth == 1))/num_positive_examples)).tolist()
                FNED["FNED after bias reduction"] += torch.abs(FNR["after_bias_reduction"]-(torch.sum(torch.logical_and(pred_after_bias_reduction == 0, ground_truth == 1))/num_positive_examples)).tolist()
                
                TPED["TPED before bias reduction"] += torch.abs(TPR["before_bias_reduction"]-(torch.sum(torch.logical_and(pred_before_bias_reduction == 1, ground_truth == 1))/num_positive_examples)).tolist()
                TPED["TPED after bias reduction"] += torch.abs(TPR["after_bias_reduction"]-(torch.sum(torch.logical_and(pred_after_bias_reduction == 1, ground_truth == 1))/num_positive_examples)).tolist()
  
                TNED["TNED before bias reduction"] += torch.abs(TNR["before_bias_reduction"]-(torch.sum(torch.logical_and(pred_before_bias_reduction == 0, ground_truth == 0))/num_negative_examples)).tolist()
                TNED["TNED after bias reduction"] += torch.abs(TNR["after_bias_reduction"]-(torch.sum(torch.logical_and(pred_after_bias_reduction == 0, ground_truth == 0))/num_negative_examples)).tolist()

                equal_opportunity_y_equal_0["before_bias_reduction"] = torch.abs(equal_opportunity_y_equal_0["before_bias_reduction"]) - (torch.sum(torch.logical_and(pred_before_bias_reduction == 1, ground_truth == 0))/num_negative_examples)
                equal_opportunity_y_equal_1["before_bias_reduction"] = torch.abs(equal_opportunity_y_equal_1["before_bias_reduction"]) - (torch.sum(torch.logical_and(pred_before_bias_reduction == 1, ground_truth == 1))/num_positive_examples)
                
                equal_opportunity_y_equal_0["after_bias_reduction"] = torch.abs(equal_opportunity_y_equal_0["after_bias_reduction"]) - (torch.sum(torch.logical_and(pred_after_bias_reduction == 1, ground_truth == 0))/num_negative_examples)
                equal_opportunity_y_equal_1["after_bias_reduction"] = torch.abs(equal_opportunity_y_equal_1["after_bias_reduction"]) - (torch.sum(torch.logical_and(pred_after_bias_reduction == 1, ground_truth == 1))/num_positive_examples)


            demographic_parity["demographic_parity before bias reduction"] = 1 - torch.abs(
                torch.mean(torch.argmax(y_pred_before_bias_reduction,axis = 1).double().to(device)[idxs_male])
                - torch.mean(torch.argmax(y_pred_before_bias_reduction,axis = 1).double().to(device)[idxs_female])
            ).tolist()    

            demographic_parity["demographic_parity after bias reduction"] = 1 - torch.abs(
                torch.mean(torch.argmax(y_pred_after_bias_reduction,axis = 1).double().to(device)[idxs_male])
                - torch.mean(torch.argmax(y_pred_after_bias_reduction,axis = 1).double().to(device)[idxs_female])
            ).tolist()
                
            equal_opportunity_y_equal_0["before_bias_reduction"] = 1- torch.abs(equal_opportunity_y_equal_0["before_bias_reduction"])
            equal_opportunity_y_equal_1["before_bias_reduction"] = 1- torch.abs(equal_opportunity_y_equal_1["before_bias_reduction"])
            
            equal_opportunity_y_equal_0["after_bias_reduction"] =  1- torch.abs(equal_opportunity_y_equal_0["after_bias_reduction"])     
            equal_opportunity_y_equal_1["after_bias_reduction"] =  1- torch.abs(equal_opportunity_y_equal_1["after_bias_reduction"]) 
            
            equality_of_odds["equality_of_odds before bias reduction"] = 0.5 * (equal_opportunity_y_equal_0["before_bias_reduction"] + equal_opportunity_y_equal_1["before_bias_reduction"]).tolist()
            equality_of_odds["equality_of_odds after bias reduction"] = 0.5 * (equal_opportunity_y_equal_0["after_bias_reduction"] + equal_opportunity_y_equal_1["after_bias_reduction"]).tolist()

    
        return AUC, accuracy, FPED, FNED, TPED, TNED, demographic_parity, equality_of_odds
    


# Some of the following parts are taken from
# https://towardsdatascience.com/fine-tuning-pretrained-nlp-models-with-huggingfaces-trainer-6326a4456e7b
# by Vincent Tan
def train_classifier(args):
    """
    Train a classifier to be used as our starting point for polcy gradient.
    We can either train from scratch or load a pretrained model depending on
    the user's choice.
    args:
        args: the arguments given by the user
    returns:
        model: the model that is going to be our starting point for polic
        y gradient
        tokenizer: the tokenizer used before giving the sentences to the
        classifier model
    """
    # Load the dataset
    train_dataset, val_dataset, test_dataset = data_loader(args)
    # The number of epochs afterwhich we save the model. We set it to this
    # value to only save the last model.
    checkpoint_steps = (
        int(train_dataset.__len__() / args.batch_size_pretraining)
    )

    if args.load_pretrained_classifier:

        tokenizer = BertTokenizer.from_pretrained(args.classifier_model, hidden_dropout_prob = args.tokenizer_dropout)
        model = BertForSequenceClassification.from_pretrained(
            args.model_path + str(checkpoint_steps),
            num_labels=len(set(train_dataset.labels)),
            # We only need the attention weights if we are going to analyze the results
            output_attentions=args.analyze_attention,
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
                + "_biased_best.pt",
                map_location=device,
            )
        )        

    else:
        # Define pretrained tokenizer and model
        model_name = args.classifier_model
        tokenizer = BertTokenizer.from_pretrained(model_name, hidden_dropout_prob = args.tokenizer_dropout)
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(set(train_dataset.labels)),
            # We only need the attention weights if we are going to analyze the results
            output_attentions = args.analyze_attention,
            num_hidden_layers = args.num_hidden_layers,
            hidden_dropout_prob = args.hidden_dropout,
            attention_probs_dropout_prob = args.attention_dropout,
            num_attention_heads = args.num_attention_heads
        )

        # Define Trainer parameters

        # Define Trainer
        classifier_args = TrainingArguments(
            output_dir=args.output_dir,
            evaluation_strategy="steps",
            eval_steps=checkpoint_steps,
            per_device_train_batch_size=args.batch_size_pretraining,
            per_device_eval_batch_size=args.batch_size_pretraining,
            num_train_epochs=args.num_epochs_pretraining,
            load_best_model_at_end=True,
        )
        trainer = Trainer(
            model=model,
            args=classifier_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Train pre-trained model
        trainer.train()
    return model, tokenizer
