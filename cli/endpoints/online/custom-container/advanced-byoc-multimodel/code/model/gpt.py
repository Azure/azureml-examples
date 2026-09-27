from transformers import pipeline, GPTNeoForCausalLM, GPT2Tokenizer

def get_callable_cpu(root_model_dir):
    model_dir = root_model_dir / "gpt"
    model = GPTNeoForCausalLM.from_pretrained(str(model_dir))
    tokenizer = GPT2Tokenizer(tokenizer_file = model_dir / "tokenizer_config.json",
                                              vocab_file = model_dir / "vocab.json", 
                                              merges_file = model_dir / "merges.txt")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

