import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

def generate_text(prompt, tokenizer, model, device, max_length=100, temperature=0.8):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    model_path = "gpt2"  # 또는 학습된 모델 저장 경로 (예: "./checkpoints/my_model/")
    tokenizer, model, device = load_model(model_path)

    print("LLM Inference 시작. 'exit' 입력 시 종료됩니다.")
    while True:
        prompt = input("Prompt 입력: ")
        if prompt.lower() == "exit":
            break
        output = generate_text(prompt, tokenizer, model, device)
        print(f"\n[Generated]\n{output}\n")

if __name__ == "__main__":
    main()
