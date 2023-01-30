# healthy-data-diet
Mitigating gender bias in text classification models by removing the stereotypical examples. We use the classifiers from 🤗 Hugging Face ([link](https://github.com/huggingface/transformers)). 

## How it works
Healthy data diet can be summarized as follows:

1. **Generating the counterfactual examples**:
2. **Finding the important examples for fairness using the GE score**: 
3. **Adding a pruned version of the original dataset to the important counterfactual examples**: 

<div style="text-align: center">
<img src="images/Healthy_diet_recipe.png" width="800">
<p style="text-align: center;"> <b>Figure:</b> The pipeline used in of the policy gradient approach. </p>
</div>

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
