from unsloth import FastLanguageModel
from transformers import AutoTokenizer

def load_model(model_name, max_seq_length, lora_config=None):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = "auto",  # "auto" = fp16 if available
        load_in_4bit = True,  # or False if you want fp16
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_config.get("r", 8),
        lora_alpha = lora_config.get("lora_alpha", 32),
        lora_dropout = lora_config.get("lora_dropout", 0.1),
        target_modules = lora_config.get("target_modules", ["q_proj", "v_proj"])
    )
    
    return model, tokenizer