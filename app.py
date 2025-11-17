
import streamlit as st
from utils.debate_utils import load_local_model, generate_counter_argument, summarize_neutral, suggest_improvements

# ページ設定
st.set_page_config(page_title="Hanron-Giron", layout="centered")
st.title("Hanron-Giron — 議論パートナー (オフライン)")

# モデル読み込み
llm = load_local_model("model/llama_japanese.gguf")  # ローカルモデルのパス

# ユーザー入力
user_input = st.text_area("あなたの意見（主張）を入力してください：")

if st.button("議論開始"):
    if not user_input.strip():
        st.warning("まず意見を入力してください。")
    else:
        with st.spinner("AIが考えています…"):
            counter = generate_counter_argument(user_input, llm)
            summary = summarize_neutral(user_input, llm)
            improvement = suggest_improvements(user_input, llm)

        st.subheader("💬 AIからの反対意見")
        st.write(counter)

        st.subheader("📝 中立的な要約")
        st.write(summary)

        st.subheader("💡 主張を強化する改善案")
        st.write(improvement)
