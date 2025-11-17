import os
from llama_cpp import Llama
import requests

MODEL_PATH = "model/llama_small.gguf"

# モデルがなければダウンロード
if not os.path.exists(MODEL_PATH):
    url = "https://huggingface.co/xxxx/llama_small/resolve/main/llama_small.gguf"
    os.makedirs("model", exist_ok=True)
    r = requests.get(url)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

llm = Llama(model_path=MODEL_PATH)
