import json
from torch.utils.data import Dataset
import torch
from transformers import DataCollatorWithPadding
import torch

class InstructionDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512, target_max_length=128):
        with open(file_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_max_length = target_max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['instruction']
        if item['input'].strip():
            prompt += "\n" + item['input']

        inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length'
        )
        labels = self.tokenizer(
            item['output'],
            truncation=True,
            max_length=self.max_length,  # 여기를 target_max_length에서 max_length로 변경
            padding='max_length'
        )

        labels_ids = labels['input_ids']
        labels_ids = [l if l != self.tokenizer.pad_token_id else -100 for l in labels_ids]

        return {
            "input_ids": torch.tensor(inputs['input_ids'], dtype=torch.long),
            "attention_mask": torch.tensor(inputs['attention_mask'], dtype=torch.long),
            "labels": torch.tensor(labels_ids, dtype=torch.long),
        }

class DataCollatorForSeq2SeqWithPadding:
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        batch = self.data_collator(features)

        # labels가 있을 경우 padding token -> -100 마스킹
        if "labels" in batch:
            labels = batch["labels"]
            labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

        return batch