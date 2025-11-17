import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

st.set_page_config(page_title="AI議論パートナー（オフライン）", layout="wide")

# -------------------------
# モデル読み込み
# -------------------------
@st.cache_resource
def load_text_model():
    model_path = "./tiny-gpt2"  # GitHub に同梱したフォルダ

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=120,
        temperature=0.8
    )

    return generator


st.title("AI議論パートナー（軽量オフライン版）")
st.write("OpenAI APIなし・ローカルモデルのみで反論・要約・改善案を生成します。")

# モデル読み込み
with st.spinner("モデル読み込み中…"):
    text_model = load_text_model()

st.success("モデル準備完了！")

# -------------------------
# 入力
# -------------------------
user_text = st.text_area("あなたの意見を入力してください：", height=200)

if st.button("議論を生成"):
    if not user_text.strip():
        st.warning("テキストを入力してください")
    else:
        with st.spinner("AIが考えています…"):

            def generate(prompt):
                out = text_model(prompt)
                return out[0]["generated_text"].replace(prompt, "").strip()

            # 反論
            prompt1 = f"意見: {user_text}\n\nこの意見に対して、建設的な反論を述べよ。"
            hanron = generate(prompt1)

            # 要約
            prompt2 = f"文章: {user_text}\n\nこの文章を短く要約せよ。"
            summary = generate(prompt2)

            # 改善案
            prompt3 = f"意見: {user_text}\n\nこの意見をより良い形に書き直せ。"
            improve = generate(prompt3)

        # -------------------------
        # 出力表示
        # -------------------------
        st.subheader("📌 AIの反論")
        st.write(hanron)

        st.subheader("📌 要約")
        st.write(summary)

        st.subheader("📌 改善案")
        st.write(improve)
