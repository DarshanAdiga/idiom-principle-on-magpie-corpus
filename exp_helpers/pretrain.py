import sys
import os
import argparse
from pprint import pprint

from datasets     import load_dataset

import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


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

    # Prepare the dataset
    datasets = load_dataset(
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
        return tokenizer(examples["text"], truncation=True )
    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"] )

    # Prepare the data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    # Prepare the model
    model = AutoModelForMaskedLM.from_pretrained(in_model_path)
    # Prepare the training arguments
    training_args = TrainingArguments(
        checkpoint_path,
        per_device_train_batch_size = 4,
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=float(epoch),
        save_strategy="epoch",
    )
    # Prepare the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )

    # Train the model
    print("Going to train the model")
    trainer.train()
    print("Training finished")

    # Save the final version of the model
    trainer.save_model(out_model_path)
    tokenizer.save_pretrained(out_model_path)
    print(f"Final model saved to {out_model_path}")


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

