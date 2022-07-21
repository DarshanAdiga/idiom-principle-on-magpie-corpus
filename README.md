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
| exp2 | [exp2](./notebooks/exp2) | No | Zero-shot | *BERT base (cased)* | All Context | On Going (4GPUs) |
| **exp3A_1**| [exp3A_1](./notebooks/exp3A_1) | Yes | Zero-shot | *BERT base (cased)* | *No Context* | On Going (4GPUs) |
| **exp3A_2**| [exp3A_2](./notebooks/exp3A_2) | Yes | Zero-shot | ToBeDecided | ToBeDecided | TODO |
| **exp3B_1**| [exp3B_1](./notebooks/exp3B_1) | Yes | Zero-shot | ToBeDecided | ToBeDecided | TODO |
| **exp3B_2**| [exp3B_2](./notebooks/exp3B_2) | Yes | Zero-shot | ToBeDecided | ToBeDecided | TODO |
| exp4 | [exp4](./notebooks/exp4) | ToBeDecided | One-shot | ToBeDecided | ToBeDecided | TODO |
| exp5 | [exp5](./notebooks/exp5) | ToBeDecided | Few-shot | ToBeDecided | ToBeDecided | TODO |

*> exp2 and onwards should have used XLNet architecture, used BERT because it was faster

**TODO:**
- Experiment with both 'Option-1' and 'Option-2' methods of adding single-token-representations.
- Conduct single-token-representations experiment with XLNet base model.

**Variations of exp3:**
The single-token-representations experiment has following variations:

1. `exp3A`: The single-token-representation contain randomly initialized embeddings.   
    1.1 `exp3A_1`: Uses the `option-1` method of adding single-token-representations, as described above.  
    1.2 `exp3A_2`: Uses the `option-2` method of adding single-token-representations, as described above.  

2. `exp3B`: The model with single-token-representation is trained on Common Crawl News Dataset (as described in the *AStitchInLanguageModels* paper).  
    2.1 `exp3B_1`: TODO  
    2.2 `exp3B_1`: TODO  

## Results

| Experiment | Dev Accuracy | Dev F1 | Test Accuracy | Test F1 |
|:-----------|:-------------|:-------|:--------------|:--------|
| exp0 | 85.16 | 83.00 | 0.0 | 0.0 |
| exp1 | 87.60 | 85.38 | 0.0 | 0.0 |

Approximate Training (Wallclock) time per experiment:
- BERT base-cased (3 GPUs): ~1.5 hours
- XLNet base-cased (4 GPUs): ~1.76 hours


# TODO
- Track & Visualise Training progress

## References
[1] Hessel Haagsma, Johan Bos, and Malvina Nissim. 2020. MAGPIE: A Large Corpus of Potentially Idiomatic Expressions. In Proceedings of the 12th Language Resources and Evaluation Conference, pages 279–287, Marseille, France. European Language Resources Association.

[2] H. Tayyar Madabushi, E. Gow-Smith, C. Scarton, and A. Villavicencio, “AStitchInLanguageModels: Dataset and Methods for the Exploration of Idiomaticity in Pre-Trained Language Models,” in Findings of the Association for Computational Linguistics: EMNLP 2021, Punta Cana, Dominican Republic, 2021, pp. 3464–3477. doi: 10.18653/v1/2021.findings-emnlp.294.

## License
TODO