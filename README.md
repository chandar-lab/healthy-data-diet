# Deep Learning on a Healthy Data Diet: Finding Important Examples for Fairness (AAAI 2023)
Mitigating gender bias in text classification models by removing the stereotypical examples ([paper](https://arxiv.org/pdf/2211.11109.pdf)). We use the classifiers from 🤗 Hugging Face ([link](https://github.com/huggingface/transformers)). 



## How it works
Healthy data diet can be summarized as follows:

1. **Generating the counterfactual examples**
2. **Finding the important examples for fairness using the GE score**
3. **Adding a pruned version of the original dataset to the important counterfactual examples** 

<div style="text-align: center">
<img src="images/Healthy_diet_recipe.png" width="400">
<p style="text-align: center;"> </p>
</div>

## Running the experiments

Follow the instructions in the `healthy_data_diet_AAAI23.ipynb` file to run the experiments in the paper.

## Citation
```
@article{zayed2022deep,
  title={Deep Learning on a Healthy Data Diet: Finding Important Examples for Fairness},
  author={Zayed, Abdelrahman and Parthasarathi, Prasanna and Mordido, Goncalo and Palangi, Hamid and Shabanian, Samira and Chandar, Sarath},
  journal={arXiv preprint arXiv:2211.11109},
  year={2022}
}
