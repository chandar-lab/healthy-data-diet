import pandas as pd
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
from models.classifier import measure_performance_metrics, Dataset

# Some of the following parts are taken from https://towardsdatascience.com/fine-tuning-pretrained-nlp-models-with-huggingfaces-trainer-6326a4456e7b by Vincent Tan
def train_baseline_1(args):
    """
    Train a baseline classifier that uses data augmentation to create a bigger dataset by flipping the gender.
    args:
        args: the arguments given by the user
    returns:
        model: the model that is going to for prediction
        tokenizer: the tokenizer used before giving the sentences to the classifier model
    """
    # Read data
    data_train = pd.read_csv("./data/" + args.dataset + "_train_original_gender.csv")
    data_valid = pd.read_csv("./data/" + args.dataset + "_valid_original_gender.csv")
    data_train_gender_swap = pd.read_csv("./data/" + args.dataset + "_train_gender_swap.csv")
    data_valid_gender_swap = pd.read_csv("./data/" + args.dataset + "_valid_gender_swap.csv")    
    
    data_train_gender_swap = data_train_gender_swap.rename(columns={data_train_gender_swap.columns[0]: data_train.columns[0]})
    data_train = pd.concat([data_train,data_train_gender_swap],axis = 0, ignore_index=True)   
    
    data_valid_gender_swap = data_valid_gender_swap.rename(columns={data_valid_gender_swap.columns[0]: data_valid.columns[0]})
    data_valid = pd.concat([data_valid,data_valid_gender_swap],axis = 0, ignore_index=True)       
    
    # The number of epochs afterwhich we save the model. We set it to this value to only save the last model.
    checkpoint_steps = (
        int(len(data_train) / args.batch_size_classifier) * args.num_epochs_classifier
    )


    # Define tokenizer and model
    model_name = args.classifier_model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(data_train[data_train.columns[1]].unique()),
        output_attentions=True,

    )

    # ----- 1. Preprocess data -----#
    # Preprocess data
    X_train = list(data_train[data_train.columns[0]])
    y_train = list(data_train[data_train.columns[1]])
    X_val = list(data_valid[data_valid.columns[0]])
    y_val = list(data_valid[data_valid.columns[1]])
    X_train_tokenized = tokenizer(
        X_train, padding=True, truncation=True, max_length=args.max_length
    )
    X_val_tokenized = tokenizer(
        X_val, padding=True, truncation=True, max_length=args.max_length
    )

    train_dataset = Dataset(X_train_tokenized, y_train)
    val_dataset = Dataset(X_val_tokenized, y_val)

    # Define Trainer parameters

    # Define Trainer
    # We divide the batch size by 2 because we now have twice the number of examples, so we do that in order not to overload the RAM.
    classifier_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=checkpoint_steps,
        per_device_train_batch_size=int(args.batch_size_classifier),
        per_device_eval_batch_size=int(args.batch_size_classifier),
        num_train_epochs=args.num_epochs_classifier,
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
