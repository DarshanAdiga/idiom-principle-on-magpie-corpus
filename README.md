# Idiom Principle on MAGPIE dataset

**Objective**: Run the Task-1 Subtask-A of [AStitchInLanguageModels](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels#task-1-idiomaticity-detection) on [MAGPIE](https://github.com/hslh/magpie-corpus) dataset

## Dataset used
Original dataset is available: [MAGPIE_filtered_split_{*}.jsonl](https://github.com/hslh/magpie-corpus).

## Experiment Setup
The original source code that runs both training and evaluation is obtained from [here](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/blob/main/Dataset/Task2/Utils/run_glue_f1_macro.py). The local copy of this code is [run_glue_f1_macro.py](./exp_helpers/run_glue_f1_macro.py).

**Notes on Reproducibility:**
1. The paths used in the notebooks are relative. Run every notebook from its own current directory.


### Experiment Tracker

| Experiment | Notebook | Single Token Rep | Dataset  | Model | Context | Status |
|:-----------|:---------|:-----------------|:---------|:------|:--------|:-------|
| exp0 | [exp0.ipynb](./experiments/exp0) | No | Zero-shot | BERT base (cased) | No | On Going |
| exp1 | [exp1.ipynb](./notebooks/exp1) | No | Zero-shot | XLNet base (cased) | Yes | On Going | 
| exp2 | [exp2.ipynb](./notebooks/exp2) | No | Zero-shot | BERT base (cased) | Previous 2 | TODO |
| exp3 | [exp3.ipynb](./notebooks/exp3) | No | Zero-shot | XLNet base (cased) | Previous 2 | TODO |
| exp4 | [exp4.ipynb](./notebooks/exp4) | No | Zero-shot | BERT base (cased) | Next 2 | TODO |
| exp5 | [exp5.ipynb](./notebooks/exp5) | No | Zero-shot | XLNet base (cased) | Next 2 | TODO |
| exp6 | [exp6.ipynb](./notebooks/exp6) | No | Zero-shot | BERT base (cased) | All Context | TODO |
| exp7 | [exp7.ipynb](./notebooks/exp7) | No | Zero-shot | XLNet base (cased) | All Context | TODO |
| exp8 | [exp8.ipynb](./notebooks/exp8) | No | One-shot | ToBeDecided | ToBeDecided | TODO |
| exp9 | [exp9.ipynb](./notebooks/exp9) | No | Few-shot | ToBeDecided | ToBeDecided | TODO |
| exp10 | [exp10.ipynb](./notebooks/exp10) | Yes | ToBeDecided | ToBeDecided | ToBeDecided | TODO |

## Results

| Experiment | Dev Accuracy | Dev F1 | Test Accuracy | Test F1 |
|:-----------|:-------------|:-------|:--------------|:--------|
| exp0 | 0.0 | 0.0 | 85.7 | 83.8 |
| exp1 | 0.0 | 0.0 | 0 | 0 |


TODO: Update this
Approximate Training time per experiment:
- BERT base (cased): ~2 hours
- XLNet base (cased): ~1 hour


# TODO
- Track & Visualise Training progress

## References
[1] Hessel Haagsma, Johan Bos, and Malvina Nissim. 2020. MAGPIE: A Large Corpus of Potentially Idiomatic Expressions. In Proceedings of the 12th Language Resources and Evaluation Conference, pages 279–287, Marseille, France. European Language Resources Association.

[2] H. Tayyar Madabushi, E. Gow-Smith, C. Scarton, and A. Villavicencio, “AStitchInLanguageModels: Dataset and Methods for the Exploration of Idiomaticity in Pre-Trained Language Models,” in Findings of the Association for Computational Linguistics: EMNLP 2021, Punta Cana, Dominican Republic, 2021, pp. 3464–3477. doi: 10.18653/v1/2021.findings-emnlp.294.

## License
TODO