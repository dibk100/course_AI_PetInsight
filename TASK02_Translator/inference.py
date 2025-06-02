import torch
from model import load_model

def generate_text(prompt, model, tokenizer, device, max_new_tokens=100):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,           # 샘플링 활성화(더 자연스러운 생성)
            top_p=0.95,               # nucleus sampling 확률
            temperature=0.8,          # 생성 다양성 조절
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="LLaMA 3 LoRA Inference with Unsloth")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML")
    parser.add_argument('--prompt', type=str, required=True, help="Input prompt for generation")
    parser.add_argument('--max_new_tokens', type=int, default=100, help="Max tokens to generate")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model(
        config['model_name'],
        max_seq_length=config.get('max_seq_length', 2048),
        lora_config=config.get('lora', {})
    )

    model.to(device)

    output_text = generate_text(args.prompt, model, tokenizer, device, max_new_tokens=args.max_new_tokens)
    print("=== Generated Text ===")
    print(output_text)

if __name__ == "__main__":
    main()
