import argparse
from train import train_model
from eval import evaluate_model
import yaml

def main():
    parser = argparse.ArgumentParser(description="LLaMA 3 Instruction Tuning with LoRA and wandb")
    parser.add_argument('--mode', choices=['train', 'eval'], required=True)
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.mode == 'train':
        train_model(config)
    else:
        evaluate_model(config)

if __name__ == "__main__":
    main()