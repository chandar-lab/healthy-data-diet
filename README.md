# reducing-gender-bias-using-auxiliary-loss
Reduce gender bias in machine learning models using an auxiliary loss.

## What is it?
We introduce a method that reduces the bias in classifiers using an auxiliary loss term. The method uses classifiers from 🤗 Hugging Face ([link](https://github.com/huggingface/transformers)). 

## How it works
Reducing the bias in classifiers using policy gradient can be summarized as follows:

1. **Training a vanilla classifier**: Train a classifier on a certain task (currently we have tested the algorithm on sexism and toxicity detection, but it should work on any other NLP classification task).
2. **Fine-tuning**: Starting with the pre-trained model, we fine-tune the weights using our auxiliary loss term. 
3. **Evaluation metrics**: To prove the efficacy of our approach, we use already existing metrics to measure the bias before and after applying policy gradient algorithm. The metrics used are: false negative equality difference (FNED) and false positive equality difference (FPED).

## Installation

### Python package
Clone the repository, then install the required packages as follows:

`pip -r install requirements.txt`

## Running the experiments

To run the experiment with Vanilla policy gradient, simply type:

`python main.py --num_epochs 30 --PG_lambda 0.5 `


## Analyzing the results

To be able to understand the results, we focus on:

1. **Attention weights**: We log the top 5 tokens to which the classification token (CLS) attends before and after de-biasing.
2. **Type of examples**: We follow the procedure in https://arxiv.org/pdf/2009.10795.pdf where the examples
    are categorized into "easy-to-learn", "hard-to-learn" and "ambiguous". The intuition is to know which category is mostly affected by the de-biasing algorithm.


To perform this analysis, simply type:

`python analyze_results.py --num_epochs_classifier 5 `
