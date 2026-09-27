from transformers import AutoTokenizer, OPTForCausalLM, pipeline 

def get_callable_cpu(root_model_dir):
    model_dir = root_model_dir / "opt"
    model = OPTForCausalLM.from_pretrained(str(model_dir)) 
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe