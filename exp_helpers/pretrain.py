import sys
import os
import argparse
from pprint import pprint
import logging

import datasets
from datasets     import load_dataset

import torch
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, TrainerCallback

import pynvml

class GPU_Helper(TrainerCallback):
    """An subclass of TrainerCallback that prints the GPU utilization every epoch."""

    def __init__(self):
        # First, we need to initialize the library
        pynvml.nvmlInit() # Fail, if no Nvidia driver found

        self.num_devices = pynvml.nvmlDeviceGetCount()
        # Ref: https://huggingface.co/docs/transformers/perf_train_gpu_one
        print('#'*40)
        print(f"[GPU_Helper] Number of devices: {self.num_devices}")
        print('#'*40)

    def print_gpu_utilization(self):
        for dev_id in range(self.num_devices):
            print(f"[GPU_Helper] GPU {dev_id} utilization:")
            handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used = mem_info.used//1024**2
            total = mem_info.total // (1024 ** 2)
            print(f">>GPU memory occupied (used/total): {used}/{total} MB.")

    def print_summary(self, result):
        print(f"Time: {result.metrics['train_runtime']:.2f}")
        print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
        self.print_gpu_utilization()

    def on_epoch_end(self, args, state, control, **kwargs):
        """Overridden method of TrainerCallback"""
        print(f"[GPU_Helper] Epoch:{state.epoch}. Completed {state.global_step}/{state.max_steps} steps.")
        self.print_gpu_utilization()
        sys.stdout.flush()

def train_and_save(in_model_path, training_data_dir, checkpoint_path, out_model_path, epoch):
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint directory {checkpoint_path} already exists. Exiting.")
        sys.exit(1)
    os.makedirs(checkpoint_path)

    if os.path.exists(out_model_path):
        print(f"Model directory {out_model_path} already exists. Exiting.")
        sys.exit(1)
    os.makedirs(out_model_path)

    # Set the logging
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger = logging.getLogger(__name__)
    log_level = logging.DEBUG
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Device configuration
    device = torch.device('cpu')
    gpu_helper = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_helper = GPU_Helper()
        gpu_helper.print_gpu_utilization()

    print(f"### Device: {device}")

    # Prepare the dataset
    raw_datasets = load_dataset(
        "text",
        data_files = {
            "train"     : os.path.join(training_data_dir, "train.txt"),
            "validation": os.path.join(training_data_dir, "eval.txt"),
        }
    )

    # Prepare the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        in_model_path, 
        use_fast = False,
        max_length = 510,
        force_download = True,
    )
    # Tokenize the datasets
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, batch_size=50000, num_proc=28, remove_columns=["text"] )

    # Prepare the data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    # Prepare the model
    model = AutoModelForMaskedLM.from_pretrained(in_model_path)

    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"### [Before] Model.Config.vocab_size: {model.config.vocab_size}")
    model.config.vocab_size = len(tokenizer)
    logger.info(f"### [After] Model.Config.vocab_size: {model.config.vocab_size}")

    # Prepare the training arguments
    training_args = TrainingArguments(
        checkpoint_path,
        per_device_train_batch_size = 8,
        per_device_eval_batch_size = 8,
        gradient_accumulation_steps=32,
        evaluation_strategy = "steps",
        eval_steps = 905, # These many steps should mean 1 epoch (considering grad accumulation)
        logging_strategy = "steps",
        logging_steps = 905, # These many steps should mean 1 epoch (considering grad accumulation)
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=float(epoch),
        save_strategy="epoch",
        debug="underflow_overflow",
    )
    # Prepare the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )

    logger.info(f"### PARALLEL_MODEL: {training_args.parallel_mode}")

    # If GPU is available,
    if gpu_helper:
        # Add the GPU utilization callback
        trainer.add_callback(gpu_helper)
        # Print the current GPU utilization
        gpu_helper.print_gpu_utilization()
        sys.stdout.flush()

    # Train the model
    print("Going to train the model")
    train_result = trainer.train()
    print("Training finished")

    # Save the final version of the model
    trainer.save_model(out_model_path)
    tokenizer.save_pretrained(out_model_path)
    print(f"Final model saved to {out_model_path}")

    # Save the metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


# Define main function with argsparser
if __name__ == "__main__":
    # Take arguments from command line using argparse
    parser = argparse.ArgumentParser(description='Train a model & save it')
    parser.add_argument('--in_model_path', type=str, help='Model (updated with idiom tokens) to train')
    parser.add_argument('--training_data_dir', type=str, help='Directory containing train.txt & eval.txt datasets')
    parser.add_argument('--checkpoint_path', type=str, help='Path to save the intermediate checkpoints')
    parser.add_argument('--out_model_path', type=str, help='Path to save the final model')
    parser.add_argument('--epoch', type=int, help='Training Epochs', default=5)

    args = parser.parse_args()
    pprint(f"Training Arguments: {args}")

    train_and_save(args.in_model_path, args.training_data_dir, args.checkpoint_path, args.out_model_path, args.epoch)

