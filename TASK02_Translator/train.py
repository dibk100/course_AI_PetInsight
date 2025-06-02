from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer
from dataset import InstructionDataset
import wandb
from model import load_model

def train_model(config):
    wandb.init(project=config['wandb_project'])

    model, tokenizer = load_model(
        config['model_name'],
        max_seq_length=config.get("max_seq_length", 2048),
        lora_config=config.get('lora', {})
    )

    train_dataset = InstructionDataset(config['train_file'], tokenizer)

    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        per_device_train_batch_size=config['batch_size'],
        num_train_epochs=config['num_epochs'],
        logging_dir=config['logging_dir'],
        save_steps=config.get('save_steps', 100),
        logging_steps=config.get('logging_steps', 10),
        fp16=True,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()