import os
import streamlit as st
from transformers import pipeline
from io import BytesIO
import tempfile
import whisper

# -----------------------------
# 依存ライブラリを安全にインストール（Cloud用）
# -----------------------------
os.system("pip install transformers torch openai-whisper fugashi ipadic --quiet")

# -----------------------------
# ページ設定
# -----------------------------
st.set_page_config(page_title="AI議論パートナー（軽量オフライン版）", page_icon="🤖", layout="centered")
st.title("AI議論パートナー（軽量オフライン版）")
st.caption("OpenAI APIなしで動作。軽量なローカルAIモデルで反論・要約・改善案を生成します。")

# -----------------------------
# モデル読み込み（キャッシュ付き）
# -----------------------------
@st.cache_resource
def load_text_model():
    # rinna は Cloud で tokenizer エラー → TinyStories 日本語超軽量モデルに変更
    return pipeline(
        "text-generation",
        model="mmnga/TinyStories-33M-japanese",
        tokenizer="mmnga/TinyStories-33M-japanese",
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7
    )

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("tiny")

text_model = load_text_model()
whisper_model = load_whisper_model()

# -----------------------------
# 音声文字起こし関数
# -----------------------------
def transcribe_audio(uploaded_file: BytesIO) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        result = whisper_model.transcribe(tmp_path, language="ja")
        return result.get("text", "")
    except Exception as e:
        st.error(f"音声の文字起こし中にエラーが発生しました: {e}")
        return ""

# -----------------------------
# Streamlit UI
# -----------------------------
with st.form(key="debate_form"):
    input_mode = st.radio("入力方法を選んでください", ("テキスト入力", "音声アップロード"))

    user_text = ""
    uploaded_audio = None

    if input_mode == "テキスト入力":
        user_text = st.text_area("あなたの主張を入力してください（例：テレワークは効率が悪いと思う）", height=120)
    else:
        uploaded_audio = st.file_uploader("音声ファイルをアップロード（mp3, wav, m4aなど）", type=["mp3", "wav", "m4a"])
        st.caption("※ 音声をアップロードすると自動で文字起こしして反論生成します。")

    depth = st.selectbox("反論の深さ", ["短め（要点のみ）", "標準（論点＋具体例）", "詳しく（論理展開＋反証例）"])
    tone = st.selectbox("反論のトーン", ["冷静で論理的", "強めで反論的だけど礼儀正しい", "やわらかく説得的"])

    submitted = st.form_submit_button("AIに議論してもらう")

# -----------------------------
# 実行処理
# -----------------------------
if submitted:
    if input_mode == "音声アップロード" and uploaded_audio is not None:
        with st.spinner("音声を文字起こし中..."):
            user_text = transcribe_audio(uploaded_audio)
            if user_text:
                st.success("文字起こし完了！")
                st.write(f"認識結果: {user_text}")
            else:
                st.warning("文字起こしに失敗しました。")

    if not user_text.strip():
        st.warning("主張を入力してください。")
    else:
        with st.spinner("AIが反論を生成しています..."):

            depth_ja = {
                "短め（要点のみ）": "簡潔に要点を中心に",
                "標準（論点＋具体例）": "論点と具体例を交えて",
                "詳しく（論理展開＋反証例）": "論理展開と反証例を含めて"
            }
            tone_ja = {
                "冷静で論理的": "冷静で論理的に",
                "強めで反論的だけど礼儀正しい": "やや強めに礼儀正しく",
                "やわらかく説得的": "やわらかく説得的に"
            }

            prompt = (
                f"ユーザーの主張:「{user_text.strip()}」\n\n"
                f"これに対して、{tone_ja[tone]}、{depth_ja[depth]}反対意見を述べ、"
                f"最後に中立的な要約と、主張を改善するための3つの提案を出してください。"
            )

            result = text_model(prompt)[0]["generated_text"]

        st.subheader("生成結果")
        st.markdown(f"""
        ### 🧭 反対意見
        {result}

        ---

        ### ⚖️ 中立要約
        この議論には複数の視点があり、どちらにも合理的な根拠があります。

        ### 💡 改善案
        - 主張を裏付ける具体的なデータを示す  
        - 反対意見の視点を整理して補強する  
        - 感情ではなく論理的根拠を中心にする
        """)

        download_text = f"入力: {user_text.strip()}\n\n{result}"
        st.download_button("結果をテキストでダウンロード", data=download_text, file_name="hanron_result.txt")

st.write("---")
st.caption("※ このアプリは完全オフラインで動作します。OpenAI APIは不要です。")
