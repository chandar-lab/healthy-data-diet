from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from models.data_loader import data_loader
import torch
import json
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions.categorical import Categorical
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def measure_bias_metrics(model_after_bias_reduction, dataset, args,run):
    """
    Compute metrics that measure the bias and performance before and after applying our
    de-biasing algorithm on the validation dataset.
    The metrics that we compute are the following:
    1) Independence
    2) Accuracy 
    args:
        model_after_bias_reduction: the model after updating its weights based for bias reduction
        dataset: the dataset on which the metrics are measured, which is the validation dataset
        args: the arguments given by the user
        run : the index of the current run, that goes from 0 to the number of runs defined by the user
    returns:
        the function doesnt return anything, since all the metrics are saved
        in json files.
    """
    independence = {}
    accuracy = {}

    checkpoint_steps = (
        int(
            len(pd.read_csv("./data/" + args.dataset + "_train_original_gender.csv"))
            / args.batch_size_pretraining
        )
        * args.num_epochs_pretraining
    )

    # Load trained model before bias reduction
    model_path = "./saved_models/checkpoint-" + str(checkpoint_steps)
    model_before_bias_reduction = BertForSequenceClassification.from_pretrained(
        model_path, num_labels=len(set(dataset.labels)), output_attentions=True
    ).to(device)      
    
    num_labels = len(set(dataset.labels))
    # We create empy numpy arrays which will contain the predictions of the model
    y_pred_original_gender_after_bias_reduction =  torch.ones([0,num_labels]).to(device)
    y_pred_original_gender_before_bias_reduction =  torch.ones([0,num_labels]).to(device)
    
    y_pred_opposite_gender_after_bias_reduction =  torch.ones([0,num_labels]).to(device)
    y_pred_opposite_gender_before_bias_reduction =  torch.ones([0,num_labels]).to(device)    

    with torch.no_grad():
        
        for i in range(int(np.ceil(len(dataset) / args.batch_size))):
    
            results_original_gender_after_bias_reduction = model_after_bias_reduction.forward(
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
            results_gender_swap_after_bias_reduction = model_after_bias_reduction.forward(
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
    
            results_original_gender_before_bias_reduction = model_before_bias_reduction.forward(
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
            results_gender_swap_before_bias_reduction = model_before_bias_reduction.forward(
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
                   
            # Get the predictions of the new batch
            batch_original_gender = results_original_gender_after_bias_reduction
            # Add them to the total predictions
            y_pred_original_gender_after_bias_reduction = torch.cat((y_pred_original_gender_after_bias_reduction, batch_original_gender), 0)
    
            # Do the same for the sentences after gender swapping
            batch_gender_swap = results_gender_swap_after_bias_reduction
            y_pred_opposite_gender_after_bias_reduction = torch.cat((y_pred_opposite_gender_after_bias_reduction, batch_gender_swap), 0)        
      
            # Get the predictions of the new batch
            batch_original_gender = results_original_gender_before_bias_reduction
            # Add them to the total predictions
            y_pred_original_gender_before_bias_reduction = torch.cat((y_pred_original_gender_before_bias_reduction, batch_original_gender), 0)
    
            # Do the same for the sentences after gender swapping
            batch_gender_swap = results_gender_swap_before_bias_reduction
            y_pred_opposite_gender_before_bias_reduction = torch.cat((y_pred_opposite_gender_before_bias_reduction, batch_gender_swap), 0)            
          
        # The kl divergence is the average between the kl div with and without gender flipping (more details in https://guide.allennlp.org/fairness#2)
        independence["after_bias_reduction"] = torch.mean(kl_divergence(Categorical(F.softmax(y_pred_original_gender_after_bias_reduction[dataset.gender_swap],1)), Categorical(F.softmax(y_pred_opposite_gender_after_bias_reduction[dataset.gender_swap],1))))
        
        # we repeat the same procedure for the model before bias reduction
        independence["before_bias_reduction"] = torch.mean(kl_divergence(Categorical(F.softmax(y_pred_original_gender_before_bias_reduction[dataset.gender_swap],1)), Categorical(F.softmax(y_pred_opposite_gender_before_bias_reduction[dataset.gender_swap],1))))
        
        output_file = "./output/independence_" + args.method + "_" + args.approach + "_" + args.dataset + "_run_" + str(run+1) + ".json"
        with open(output_file, "w+") as f:
            json.dump(str(independence), f, indent=2)    

        # ===================================================#
        # Here we calculate the accuracy
                     
        accuracy["before_bias_reduction"] = torch.sum(torch.argmax(y_pred_original_gender_after_bias_reduction,axis = 1).to(device) == torch.tensor(dataset.labels).to(device))/len(dataset.labels)
        accuracy["after_bias_reduction"] = torch.sum(torch.argmax(y_pred_original_gender_before_bias_reduction,axis = 1).to(device) == torch.tensor(dataset.labels).to(device))/len(dataset.labels)
        
        output_file = "./output/validation_accuracy_" + args.method + "_" + args.approach + "_" + args.dataset + "_run_" + str(run+1) + ".json"
        with open(output_file, "w+") as f:
            json.dump(str(accuracy), f, indent=2)     
            
# Some of the following parts are taken from
# https://towardsdatascience.com/fine-tuning-pretrained-nlp-models-with-huggingfaces-trainer-6326a4456e7b
# by Vincent Tan
def train_classifier(args,data_augmentation_flag=None):
    """
    Train a classifier to be used as our starting point for polcy gradient.
    We can either train from scratch or load a pretrained model depending on
    the user's choice.
    args:
        args: the arguments given by the user
        data_augmentation_flag: a flag to choose whether or not to apply data
        augmentation, meaning that the number of examples doubles because we
        flip the gender in each example and add it as a new example.
    returns:
        model: the model that is going to be our starting point for polic
        y gradient
        tokenizer: the tokenizer used before giving the sentences to the
        classifier model
    """
    # Load the dataset
    train_dataset, val_dataset, test_dataset = data_loader(args,apply_data_augmentation=data_augmentation_flag)
    # The number of epochs afterwhich we save the model. We set it to this
    #value to only save the last model.
    checkpoint_steps = (
        int(train_dataset.__len__()/ args.batch_size_pretraining) * args.num_epochs_pretraining
    )

    if args.load_pretrained_classifier:

        tokenizer = BertTokenizer.from_pretrained(args.classifier_model)
        model = BertForSequenceClassification.from_pretrained(
            args.model_path + str(checkpoint_steps),
            num_labels=len(set(train_dataset.labels)),
            output_attentions=True,
        )

    else:
        # Define pretrained tokenizer and model
        model_name = args.classifier_model
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(set(train_dataset.labels)),
            output_attentions=True,
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
            compute_metrics=measure_performance_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Train pre-trained model
        trainer.train()
    return model, tokenizer


def measure_performance_metrics(p):
    pred, labels = p
    pred = np.argmax(pred[0], axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average="micro")
    precision = precision_score(y_true=labels, y_pred=pred, average="micro")
    f1 = f1_score(y_true=labels, y_pred=pred, average="micro")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

