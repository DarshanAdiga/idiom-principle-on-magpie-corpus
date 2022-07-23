# Idiom Principle on MAGPIE dataset

**Objective**: Run the Task-1 Subtask-A of [AStitchInLanguageModels](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels#task-1-idiomaticity-detection) on [MAGPIE](https://github.com/hslh/magpie-corpus) dataset

## Dataset used
Original dataset is available: [MAGPIE_filtered_split_{*}.jsonl](https://github.com/hslh/magpie-corpus).

## Experiment Setup
- The original source code that runs both training and evaluation is obtained from [here](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/Utils/run_glue_f1_macro.py). The local copy of this code is [run_glue_f1_macro.py](./exp_helpers/run_glue_f1_macro.py).

**Notes on Reproducibility:**
1. The paths used in the notebooks are relative. Run every notebook from its own current directory.
2. It is better to use even-numbered GPUs (2 is slow, 4 is better) for training & evaluation. Specifically, the batch size should be divisible by number of GPUs.

### Single Token Representation
The code for adding single-token-representations is based on:
1. [Adding new Tokens](https://huggingface.co/transformers/v2.11.0/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.add_tokens)
2. [Manual method of adding tokens](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#adding-idiom-tokens-to--transformers-models)

**Variations:**  
The MAGPIE dataset contains `idiom` column but the sentences can contain different *surface form* of those idioms(due discontiguity & variations of MWEs). Approximately 50% of the sentences contain a different form than the given `idiom` column. Thus, two different ways of adding single-token-representations are used:

* Option-1: Just convert the values in `idiom` column to tokens, irrespective of how they are used in the sentence. In other words, this approach will make the LM model to learn only those tokens which have an exact match.

* Option-2: Use the `offsets` column and extract the actual MWE from the sentence. This will capture all possible MWEs in the data, but the number of unique tokens would be very high


### Experiment Tracker

| Experiment | Notebook | Single Token Rep | Dataset  | Model | Context | Status |
|:-----------|:---------|:-----------------|:---------|:------|:--------|:-------|
| exp0 | [exp0](./experiments/exp0) | No | Zero-shot | BERT base (cased) | No Context | Done (3GPUs) |
| exp1 | [exp1](./notebooks/exp1) | No | Zero-shot | XLNet base (cased) | No Context | Done (4GPUs) | 
| exp2 | [exp2](./notebooks/exp2) | No | Zero-shot | *BERT base (cased)* | All Context | Done (4GPUs) |
| **exp3A_1**| [exp3A_1](./notebooks/exp3A_1) | Yes | Zero-shot | *BERT base (cased)* | No Context | Done (4GPUs) |
| **exp3A_2**| [exp3A_2](./notebooks/exp3A_2) | Yes | Zero-shot | *BERT base (cased)* | No Context | Done (4GPUs) |
| **exp3B_1**| [exp3B_1](./notebooks/exp3B_1) | Yes | Zero-shot | ToBeDecided | ToBeDecided | TODO |
| **exp3B_2**| [exp3B_2](./notebooks/exp3B_2) | Yes | Zero-shot | ToBeDecided | ToBeDecided | TODO |
| exp4 | [exp4](./notebooks/exp4) | ToBeDecided | One-shot | ToBeDecided | ToBeDecided | TODO |
| exp5 | [exp5](./notebooks/exp5) | ToBeDecided | Few-shot | ToBeDecided | ToBeDecided | TODO |

*> exp2 and onwards should have used XLNet architecture, used BERT because it was faster

**TODO:**
- Conduct single-token-representations experiment with XLNet base model.
- The *AStitchInLanguageModels* paper does Idiom-includ/exclude experiment as well in Task-1. Try that as well, if required.

**Variations of exp3:**
In both of the below experiments (exp3A and exp3B), the MWEs are replaced by their corresponding single tokens in the training data. 
The single-token-representations experiment has following variations:

1. `exp3A`: The single-token-representation contain randomly initialized embeddings.   
    1.1 `exp3A_1`: Uses the `option-1` method of adding single-token-representations, as described above.  
    1.2 `exp3A_2`: Uses the `option-2` method of adding single-token-representations, as described above.  

2. `exp3B`: The model with single-token-representation is first trained(fine-tuned) with a Masked-LM objective on Common Crawl News Dataset (as described in the *AStitchInLanguageModels* paper). The steps followed [here](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#generating-pre-training-data) are taken as reference.  
**Steps:**  

    i. Add the new tokens to the vocabulary of the model. This leads to two variations of models using `option-1` and `option-2`.  

    ii. Train(fine-tune) the model with a *Masked-LM* objective on CC-News corpus. The pre-processed CC-News data for this purpose is already available [here](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#generating-pre-training-data). It has been used directly in these experiments. The training scripts are available [experiments/pretraining](experiments/pretraining).  

    iii. Use this fine-tuned model with a *SequenceClassification* objective on the MAGPIE dataset as done by previous experiments. This leads to two experiments: `exp3B_1` and `exp3B_2`.  


**TODO:**
- The pre-processed CC News Corpus used in the `step 2.ii` above has been created using [this script](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#extract-data-from-common-crawl). This method uses a list of idioms to identify sentences with and without idioms in the CC News dataset. Evaluate the percentage of overlap between the list of idioms used by this script and the list of idioms available in the MAGPIE dataset.

## Results

| Experiment | Dev Accuracy | Dev F1 | Test Accuracy | Test F1 |
|:-----------|:-------------|:-------|:--------------|:--------|
| exp0 | 85.16 | 83.00 | 0.0 | 0.0 |
| exp1 | 87.60 | 85.38 | 0.0 | 0.0 |
| exp2 | 84.91 | 81.50 | 0.0 | 0.0 |
| exp3A_1| 78.26 | 67.54 | 0.0 | 0.0 |
| exp3A_2| 80.39 | 74.21 | 0.0 | 0.0 |

Approximate Training (Wallclock) time per experiment:
- BERT base-cased (3 GPUs): ~1.5 hours
- BERT base-cased (4 GPUs): ~1.2 hours
- XLNet base-cased (4 GPUs): ~1.76 hours


# TODO
- Track & Visualise Training progress

## References
[1] Hessel Haagsma, Johan Bos, and Malvina Nissim. 2020. MAGPIE: A Large Corpus of Potentially Idiomatic Expressions. In Proceedings of the 12th Language Resources and Evaluation Conference, pages 279–287, Marseille, France. European Language Resources Association.

[2] H. Tayyar Madabushi, E. Gow-Smith, C. Scarton, and A. Villavicencio, “AStitchInLanguageModels: Dataset and Methods for the Exploration of Idiomaticity in Pre-Trained Language Models,” in Findings of the Association for Computational Linguistics: EMNLP 2021, Punta Cana, Dominican Republic, 2021, pp. 3464–3477. doi: 10.18653/v1/2021.findings-emnlp.294.

## License
TODO