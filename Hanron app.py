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
        temperature=0.8,
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
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )
    return gpt_pipeline

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("tiny")

text_model = load_text_model()
whisper_model = load_whisper_model()

# -----------------------------
# Whisper 音声文字起こし
# -----------------------------
def transcribe_audio(uploaded_file: BytesIO) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        result = whisper_model.transcribe(tmp_path, language="ja")
        return result.get("text", "")
    except Exception as e:
        st.error(f"音声の文字起こし中にエラー: {e}")
        return ""

# -----------------------------
# UIフォーム
# -----------------------------
with st.form(key="debate_form"):
    input_mode = st.radio("入力方法を選択", ("テキスト入力", "音声アップロード"))

    user_text = ""
    uploaded_audio = None

    if input_mode == "テキスト入力":
        user_text = st.text_area("あなたの主張を入力してください", height=120)
    else:
        uploaded_audio = st.file_uploader("音声ファイルをアップロード", type=["mp3", "wav", "m4a"])
        st.caption("※ アップロードすると自動で文字起こしされます")

    depth = st.selectbox("反論の深さ", ["短め", "標準", "詳しく"])
    tone = st.selectbox("反論のトーン", ["冷静で論理的", "強めで礼儀正しく", "やわらかく説得的"])

    submitted = st.form_submit_button("AIに議論してもらう")

# -----------------------------
# 実行
# -----------------------------
if submitted:
    # 音声 → テキスト
    if input_mode == "音声アップロード" and uploaded_audio is not None:
        with st.spinner("音声を文字起こし中..."):
            user_text = transcribe_audio(uploaded_audio)
            if user_text:
                st.success("文字起こし完了！")
                st.write(f"認識: {user_text}")
            else:
                st.warning("文字起こしに失敗しました")

    if not user_text.strip():
        st.warning("主張を入力してください。")
    else:
        with st.spinner("AIが反論を生成中..."):
            depth_map = {
                "短め": "簡潔に要点だけを述べて",
                "標準": "論点と具体例を交えて",
                "詳しく": "論理展開と反証例を含めて"
            }
            tone_map = {
                "冷静で論理的": "冷静で論理的に",
                "強めで礼儀正しく": "やや強めに礼儀正しく",
                "やわらかく説得的": "やわらかく説得的に"
            }

            prompt = (
                f"ユーザーの主張:「{user_text.strip()}」\n\n"
                f"{tone_map[tone]}、{depth_map[depth]}反対意見を述べ、"
                f"その後に中立的な要約を示し、"
                f"最後に主張を改善するための3つの改善案を提案してください。\n"
            )

            result = text_model(prompt)[0]["generated_text"]

        # -----------------------------
        # 表示
        # -----------------------------
        st.subheader("生成結果")
        st.markdown(f"""
        ### 🧭 反対意見
        {result}

        ---

        ### ⚖️ 中立要約
        議論には複数の視点があり、どちらにも一定の合理性があります。

        ### 💡 改善案
        - 主張を支える客観的データを追加する  
        - 反対意見への理解を示した上で主張を補強する  
        - 感情ではなく論理的根拠を中心に説明する  
        """)

        # ダウンロード
        st.download_button(
            "結果をテキストでダウンロード",
            data=f"入力: {user_text}\n\n{result}",
            file_name="hanron_result.txt"
        )

st.write("---")
st.caption("※ このアプリは完全ローカルで動作し、APIキーは不要です。")
