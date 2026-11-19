from fastapi import FastAPI, File 
from handler import Handler 
from random import randint
from pydantic import BaseModel
from anyio.lowlevel import RunVar
from multiprocessing import cpu_count
import os
from pathlib import Path 

app = FastAPI()
handler = Handler(base_model_dir=Path(os.getenv("MODEL_DIR")) / "models")

class LMJSON(BaseModel):
    value : str

@app.post("/opt")
async def opt(payload: LMJSON):
    return await handler.ainfer("opt", payload.value)

@app.post("/gpt")
async def gpt(payload: LMJSON):
    return await handler.ainfer("gpt", payload.value)

@app.get("/ready")
def health(): 
    return "Ready"

@app.get("/live")
def live(): 
    return "Live"

@app.get("/")
def root(): 
    return ""