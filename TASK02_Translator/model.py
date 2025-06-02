from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer
