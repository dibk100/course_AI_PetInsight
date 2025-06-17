import torch
from torch.utils.data import DataLoader
from model import load_model
from dataset import InstructionDataset

def evaluate_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(
        config['model_name'],
        max_seq_length=config.get('max_seq_length', 2048),
        lora_config=config.get('lora', {})
    )
    model.eval()

    eval_dataset = InstructionDataset(config['eval_file'], tokenizer)
    dataloader = DataLoader(eval_dataset, batch_size=1)

    for batch in dataloader:
        input_ids = batch['input_ids'].unsqueeze(0).to(device)
        outputs = model.generate(input_ids, max_new_tokens=100)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(decoded)