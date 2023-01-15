# Idiom Principle on MAGPIE dataset

**Objective**: Run the Task-1 Subtask-A of [AStitchInLanguageModels](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels#task-1-idiomaticity-detection) on [MAGPIE](https://github.com/hslh/magpie-corpus) dataset

## Dataset used
Original dataset is available: [MAGPIE_filtered_split_{*}.jsonl](https://github.com/hslh/magpie-corpus).

## Experiment Setup
- The original source code that runs both training and evaluation is obtained from [here](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/Utils/run_glue_f1_macro.py). The local copy of this code is [run_glue_f1_macro.py](./exp_helpers/run_glue_f1_macro.py).

**Notes on Reproducibility:**
1. The paths used in the notebooks are relative. Run every notebook from its own current directory.
2. It is better to use even-numbered GPUs (2 is slow, 4 is better) for training & evaluation. Specifically, the batch size should be divisible by number of GPUs.
3. When running the experiments on JarvisLabs.ai, follow the below steps:  
   a. Uninstall the existing version of PyTorch from the instance (it should be PyTorch 1.13)  
   b. Install the PyTorch 1.12.0 version for the correct CUDA version, using the below command:
   ```bash
   pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
   ```
   More details can be found [here](https://pytorch.org/get-started/previous-versions/)

### Single Token Representation
The code for adding single-token-representations is based on:
1. [Adding new Tokens](https://huggingface.co/transformers/v2.11.0/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.add_tokens)
2. [Manual method of adding tokens](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#adding-idiom-tokens-to--transformers-models)

**Variations:**  
The MAGPIE dataset contains `idiom` column but the sentences can contain different *surface form* of those idioms(due discontiguity & variations of MWEs). Approximately 50% of the sentences contain a different form than the given `idiom` column. Thus, two different ways of adding single-token-representations are used:

* Option-1: Just convert the values in `idiom` column to tokens, irrespective of how they are used in the sentence. In other words, this approach will make the LM model to learn only those tokens which have an exact match.

* Option-2: Use the `offsets` column and extract the actual MWE from the sentence. This will capture all possible MWEs in the data, but the number of unique tokens would be very high

### Variations of exp3:
In both of the below experiments (exp3A and exp3B), the MWEs are replaced by their corresponding single tokens in the training data. 
The single-token-representations experiment has following variations:

1. `exp3A`: The single-token-representation contain randomly initialized embeddings.   
    1.1 `exp3A_1`: Uses the `option-1` method of adding single-token-representations, as described above.  
    1.2 `exp3A_2`: Uses the `option-2` method of adding single-token-representations, as described above.  

2. `exp3B`: The model with single-token-representation is first trained(fine-tuned) with a Masked-LM objective on Common Crawl News Dataset (as described in the *AStitchInLanguageModels* paper). The steps followed [here](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#generating-pre-training-data) are taken as reference.  
**Steps:**  

    i. Add the new tokens to the vocabulary of the model. This leads to two variations of models using `option-1` and `option-2`.  

    ii. Train(fine-tune) the model with a *Masked-LM* objective on the pre-processed CC-News corpus.  

    **CC News Data Preparation:**  
    The pre-processed CC-News data for this purpose had to be generated with slight modifications. The original steps are described [here](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/README.md#generating-pre-training-data). The modified preprocessing scripts are available [here](./exp_helpers/prepare_cc_corpus/).  

    First, download and preprocess the CC News Corpus using `experiments/exp3B_1/process_cc_hpc.sh` script.  
    Then, prepare the training data for pretraining with single-tokens using `experiments/exp3B_1/create_pretrain_data_hpc.sh` script.  
    And then, split the `all_replace_data.txt` file into train & eval sets using `experiments/exp3B_1/split_pretrain_data_hpc.sh` script.  
    
    **Pre Training:**
    Finally, train the model (with updated tokens in step i.) with a *Masked-LM* objective on this data. The original pretraining-script is refered from [here](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/SubtaskA-Pre_Train/preTrain.py). The customized training scripts are available [exp3B_1/train_and_save_hpc](experiments/exp3B_1/train_and_save_hpc.sh).  

    iii. Use this fine-tuned model with a *SequenceClassification* objective on the MAGPIE dataset:
        - First convert the MaskedLM model to a SequenceClassification model using `exp3B_1/MLM_to_SeqClass_model_converter.ipynb`  
        - Then fine-tune on MAGPIE dataset as done by previous experiments(that is, using `exp3B_1/hpc.sh`).  

    iv. Follow these steps of `option-2` as well using the idioms of Option-2.  This leads to two experiments: `exp3B_1` and `exp3B_2`.  

## Experiment Tracker

| Experiment | Code | Single Token Rep | Dataset  | Model | Context | Status |
|:-----------|:---------|:-----------------|:---------|:------|:--------|:-------|
| exp0 | [exp0](./experiments/exp0) | No | Zero-shot | BERT base (cased) | No Context | Done (3GPUs) |
| exp1 | [exp1](./experiments/exp1) | No | Zero-shot | XLNet base (cased) | No Context | Done (4GPUs) | 
| exp2 | [exp2](./experiments/exp2) | No | Zero-shot | *BERT base (cased)* | All Context | Done (4GPUs) |
| **exp3A_1**| [exp3A_1](./experiments/exp3A_1) | Yes | Zero-shot | **bert-base-uncased** | No Context | Done (RTX5000 x 1) |
| **exp3A_2**| [exp3A_2](./experiments/exp3A_2) | Yes | Zero-shot | *BERT base (cased)* | No Context | Done (4GPUs) |
| **exp3B_1**| [exp3B_1](./experiments/exp3B_1) | Yes | Zero-shot | **bert-base-uncased** | No Context | Done (RTX5000 x 1) |
| **exp3B_2**| [exp3B_2](./experiments/exp3B_2) | Yes | Zero-shot | ToBeDecided | ToBeDecided | TODO |
| exp4 | [exp4](./experiments/exp4) | ToBeDecided | One-shot | ToBeDecided | ToBeDecided | TODO |
| exp5 | [exp5](./experiments/exp5) | ToBeDecided | Few-shot | ToBeDecided | ToBeDecided | TODO |

*> exp2 and onwards should have used XLNet architecture, used BERT because it was faster

**TODO:**
- Conduct single-token-representations experiment with XLNet base model.
- The *AStitchInLanguageModels* paper does Idiom-includ/exclude experiment as well in Task-1. Try that as well, if required.

## Results

| Experiment | Dev Accuracy | Dev F1 | Test Accuracy | Test F1 |
|:-----------|:-------------|:-------|:--------------|:--------|
| exp0 | 85.16 | 83.00 | 0.0 | 0.0 |
| exp1 | 87.60 | 85.38 | 0.0 | 0.0 |
| exp2 | 84.91 | 81.50 | 0.0 | 0.0 |
| *exp3A_1 | 79.23 | 71.42 | 78.33 | 71.57 |
| exp3A_2 | 80.39 | 74.21 | 0.0 | 0.0 |
| exp3B_1(*deprecated*) | 85.14 | 79.29 | 0.0 | 0.0 |
| **exp3B_1 | 82.83 | 75.66 | 81.82 | 76.55 |

- *exp3A_1 metrics are from the latest run on (RTX5000 x 1)
- **exp3B_1 metrics are from the latest run on (RTX5000 x 1)

Approximate Training (Wallclock) time per experiment:
- BERT base-cased (3 GPUs): ~1.5 hours
- BERT base-cased (4 GPUs): ~1.2 hours
- XLNet base-cased (4 GPUs): ~1.76 hours
- Pretraining BERT base-uncased on MLM task (5 GPUs): ~23 hours

- With (RTX5000 x 1) GPUs: ~1 hour 20 mins


## Error Analysis & Study
For the error analysis and to study the idiom principle, the MAGPIE PIEs are grouped into different lists based on their characteristics.  
The characteristics are observed in the MAGPIE as well as preprocessed CommonCrawl News corpus.  

The implementation of grouping of PIEs is available at [PIE_segregation_util.ipynb](./exp_helpers/PIE_segregation_util.ipynb).

The classification reports (both overall and segreated) is generated for `exp3A_1` and `exp3B_1` using the script [produce_test_results.py](./exp_helpers/produce_test_results.py).

## Statistical Significance test
The statistical significance test is done using the script [exp_helpers/statistical_significance_test.ipynb](./exp_helpers/exp_helpers/statistical_significance_test.ipynb).

Wilcoxon signed-rank test is used to test the null hypothesis that two related paired samples come from the same distribution.

**References:**
- https://pythonfordatascienceorg.wordpress.com/wilcoxon-sign-ranked-test-python/

# TODO
- Track & Visualise Training progress


## References
[1] Hessel Haagsma, Johan Bos, and Malvina Nissim. 2020. MAGPIE: A Large Corpus of Potentially Idiomatic Expressions. In Proceedings of the 12th Language Resources and Evaluation Conference, pages 279–287, Marseille, France. European Language Resources Association.

[2] H. Tayyar Madabushi, E. Gow-Smith, C. Scarton, and A. Villavicencio, “AStitchInLanguageModels: Dataset and Methods for the Exploration of Idiomaticity in Pre-Trained Language Models,” in Findings of the Association for Computational Linguistics: EMNLP 2021, Punta Cana, Dominican Republic, 2021, pp. 3464–3477. doi: 10.18653/v1/2021.findings-emnlp.294.

## License
TODO
