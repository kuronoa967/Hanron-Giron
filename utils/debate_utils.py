import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# prompts.json の読み込み
with open(os.path.join(os.path.dirname(__file__), "..", "data", "prompts.json"), "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)

# モデル名（軽量日本語 GPT 系）
MODEL_NAME = "rinna/japanese-gpt-neox-small"  # 数十MBの軽量モデル例

# トークナイザーとモデルを自動ダウンロード
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 推論関数
def model_infer(prompt: str, max_tokens: int = 256):
    out = text_generator(prompt, max_new_tokens=max_tokens, do_sample=True, top_p=0.95)
    return out[0]['generated_text']

# 各処理関数
def generate_counter_argument(text: str):
    prompt = PROMPTS["counter_argument"].replace("{input_text}", text)
    return model_infer(prompt)

def summarize_neutral(text: str):
    prompt = PROMPTS["neutral_summary"].replace("{input_text}", text)
    return model_infer(prompt)

def suggest_improvements(text: str):
    prompt = PROMPTS["improvement"].replace("{input_text}", text)
    return model_infer(prompt)
