import json
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=2048):
        with open(file_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['instruction'] + "\n" + item['input']
        target = item['output']
        input_ids = self.tokenizer(prompt, truncation=True, max_length=self.max_length, return_tensors="pt")['input_ids'][0]
        labels = self.tokenizer(target, truncation=True, max_length=self.max_length, return_tensors="pt")['input_ids'][0]
        return {"input_ids": input_ids, "labels": labels}