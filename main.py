# Main script for gathering args.

import argparse
from experiment import run_experiment

parser = argparse.ArgumentParser()


#arguments for the policy gradient algorithm
parser.add_argument('--batch_size', type=int, default=64,
                    help='Samples per batch')
parser.add_argument('--num_epochs', type=int, default=15,
                    help='Number of training epochs for the policy gradient algorithm')
parser.add_argument('--learning_rate', type=float, default=1.41e-4,
                    help='learning rate for the Bert classifier')

#arguments for the classifier
parser.add_argument('--classifier_model', choices=['bert-base-uncased'],
                    default='bert-base-uncased', help='Type of classifier used')
parser.add_argument('--dataset', choices=['Equity-Evaluation-Corpus,twitter_dataset'],
                    default='twitter_dataset', help='Type of dataset used')
parser.add_argument('--num_epochs_classifier', type=int, default=3,
                    help='Number of training epochs for the classifier')
