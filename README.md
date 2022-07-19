# Idiom Principle on MAGPIE dataset

**Objective**: Run the Task-1 Subtask-A of [AStitchInLanguageModels](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels#task-1-idiomaticity-detection) on [MAGPIE](https://github.com/hslh/magpie-corpus) dataset

## Dataset used
Original dataset is available: [MAGPIE_filtered_split_{*}.jsonl](https://github.com/hslh/magpie-corpus).

## Experiment Setup
The original source code that runs both training and evaluation is obtained from [here](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/Utils/run_glue_f1_macro.py). The local copy of this code is [run_glue_f1_macro.py](./exp_helpers/run_glue_f1_macro.py).

**Notes on Reproducibility:**
1. The paths used in the notebooks are relative. Run every notebook from its own current directory.
2. It is better to use even-numbered GPUs (2 is slow, 4 is better) for training & evaluation. Specifically, the batch size should be divisible by number of GPUs.

### Experiment Tracker

| Experiment | Notebook | Single Token Rep | Dataset  | Model | Context | Status |
|:-----------|:---------|:-----------------|:---------|:------|:--------|:-------|
| exp0 | [exp0](./experiments/exp0) | No | Zero-shot | BERT base (cased) | No Context | Done (3GPUs) |
| exp1 | [exp1](./notebooks/exp1) | No | Zero-shot | XLNet base (cased) | No Context | Done (4GPUs) | 
| exp2 | [exp2](./notebooks/exp2) | No | Zero-shot | **BERT base (cased)** | All Context | TODO (4GPUs) |
| exp3 | [exp3](./notebooks/exp3) | Yes | Zero-shot | ToBeDecided | ToBeDecided | TODO |
| exp4 | [exp4](./notebooks/exp4) | ToBeDecided | One-shot | ToBeDecided | ToBeDecided | TODO |
| exp5 | [exp5](./notebooks/exp5) | ToBeDecided | Few-shot | ToBeDecided | ToBeDecided | TODO |

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