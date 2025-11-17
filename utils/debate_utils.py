import json
from llama_cpp import Llama
import os

# プロンプト読み込み
with open(os.path.join("data", "prompts.json"), "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)

# モデルロード
def load_local_model(model_path: str, n_ctx: int = 2048):
    """ローカルの llama モデルをロード"""
    return Llama(model_path=model_path, n_ctx=n_ctx)

# 推論関数
def model_infer(llm, prompt: str, max_tokens: int = 256):
    out = llm(prompt=prompt, max_tokens=max_tokens, stop=None)
    return out["choices"][0]["text"]

def generate_counter_argument(text: str, llm):
    prompt = PROMPTS["counter_argument"].replace("{input_text}", text)
    return model_infer(llm, prompt)

def summarize_neutral(text: str, llm):
    prompt = PROMPTS["neutral_summary"].replace("{input_text}", text)
    return model_infer(llm, prompt)

def suggest_improvements(text: str, llm):
    prompt = PROMPTS["improvement"].replace("{input_text}", text)
    return model_infer(llm, prompt)
