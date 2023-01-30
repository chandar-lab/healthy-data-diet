from transformers import TrainingArguments, Trainer
from transformers import BertForSequenceClassification, RobertaForSequenceClassification
from transformers import EarlyStoppingCallback
from model.data_loader import data_loader
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_biased_classifier(
    dataset,
    CDA_examples_ranking,
    data_augmentation_ratio,
    data_diet_examples_ranking,
    data_diet_factual_ratio,
    data_diet_counterfactual_ratio,
    data_substitution_ratio,
    max_length,
    classifier_model,
    batch_size_biased_model,
    model_dir,
    use_amulet,
    num_epochs_biased_model,
    learning_rate,
    seed,
):
    """
    Train the biased classifier.
    args:
        dataset: the dataset used
        CDA_examples_ranking: the ranking of the CDA examples
        data_augmentation_ratio: The ratio of data augmentation that we apply, given that the debiasing is using data augmentation
        data_diet_examples_ranking: Type of rankings we use to pick up the examples in data pruning.
        data_diet_factual_ratio: The ratio of the factual examples that we train on while using data diet.
        data_diet_counterfactual_ratio: The ratio of the counterfactual examples that we train on while using data diet.
        data_substitution_ratio: The ratio of the dataset examples that are flipped in data substitution.
        max_length: The maximum length of the sentences that we classify (in terms of the number of tokens)
        classifier_model: the model name
        batch_size_biased_model: the batch size for the pretraining (training the biased model)
        model_dir: the Directory to the model
        use_amulet: whether or not to run the code on Amulet, which is the cluster used at Microsoft research
        num_epochs_biased_model: the number of epochs to train the biased model, which is done before bias mitigation
        learning_rate: the learning rate of the classifier
        seed: the seed used by the classifier
    returns:
        model: the model after training
    """
    # Load the dataset
    train_dataset, val_dataset, test_dataset = data_loader(
        seed,
        dataset,
        CDA_examples_ranking,
        data_augmentation_ratio,
        data_diet_examples_ranking,
        data_diet_factual_ratio,
        data_diet_counterfactual_ratio,
        data_substitution_ratio,
        max_length,
        classifier_model,
    )
    # The number of epochs afterwhich we save the model.
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

    # Define pretrained tokenizer and model
    model_name = classifier_model
    model = huggingface_model.from_pretrained(
        "./saved_models/cached_models/" + model_name,
        num_labels=len(set(train_dataset.labels)),
    )

    model = model.to(device)

    # Define Trainer parameters

    # Define Trainer
    classifier_args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="steps",
        eval_steps=checkpoint_steps,
        save_steps=checkpoint_steps,
        per_device_train_batch_size=batch_size_biased_model,
        per_device_eval_batch_size=batch_size_biased_model,
        num_train_epochs=num_epochs_biased_model,
        learning_rate=learning_rate,
        seed=seed,
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

    return model
