import json
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from model import load_model_and_tokenizer
from dataset import InstructionDataset

def train_model(config_path):
    with open(config_path) as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(config['model_name_or_path'], device)

    train_dataset = InstructionDataset(config['train_file'], tokenizer)
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        per_device_train_batch_size=config['batch_size'],
        num_train_epochs=config['num_epochs'],
        logging_dir=config['logging_dir'],
        save_steps=config.get('save_steps', 500),
        logging_steps=config.get('logging_steps', 100),
        evaluation_strategy="no",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()