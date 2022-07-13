import model 
import asyncio
from pathlib import Path 

class Handler: 
    def __init__(self, base_model_dir, models=None):
        self.models_loaded = models 
        self.model_callables = {}
        self.base_model_dir = Path(base_model_dir)
        self._load_models(models)
    
    def _load_models(self, models):
        if not models: 
            models = model.__all__
        for m in models: 
            module = getattr(model, m) 
            self.model_callables[m] = module.get_callable_cpu(self.base_model_dir)

    async def ainfer(self, model, input):
        loop = asyncio.get_running_loop() 
        res = await loop.run_in_executor(None, self.model_callables[model], input)
        return res 
    
    def infer(self, model, input):
        return self.model_callables[model](input)