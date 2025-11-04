import streamlit as st

from io import BytesIO

# -----------------------------
# 設定
# -----------------------------
# Streamlit の Secrets または環境変数で OPENAI_API_KEY を設定してください。
# 例: st.secrets["OPENAI_API_KEY"] または環境変数 OPENAI_API_KEY

OPENAI_KEY = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None
if not OPENAI_KEY:
    import os
    OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

openai.api_key = OPENAI_KEY

# -----------------------------
# ユーティリティ関数
# -----------------------------

def transcribe_audio(uploaded_file: BytesIO) -> str:
    """音声ファイルをモデル（Whisper）で文字起こしする。
    注意: 利用するモデル名やメソッドは将来変わる可能性があります。"""
    try:
        # openai の audio transcription を呼び出す
        # ここでは一般的な呼び出し例を記載します。環境によって微調整してください。
        audio_bytes = uploaded_file.read()
        audio_buffer = BytesIO(audio_bytes)
        # rewind
        audio_buffer.seek(0)
        transcript = openai.Audio.transcribe("whisper-1", audio_buffer)
        # 上の戻り値は dict や object の場合があるため柔軟に
        if isinstance(transcript, dict):
            return transcript.get("text", "")
        return str(transcript)
    except Exception as e:
        st.error(f"音声の文字起こし中にエラーが発生しました: {e}")
        return ""


def call_model(prompt: str, model: str = "gpt-4o-mini") -> str:
    """チャットモデルを呼ぶラッパー。必要に応じて system 指示や temperature を調整してください。"""
    if not openai.api_key:
        st.error("OpenAI API key が設定されていません。環境変数 OPENAI_API_KEY か Streamlit secrets に設定してください。")
        return ""
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "あなたは建設的で礼儀正しい議論相手です。相手の意見に対して反対の立場から論理的・事実ベースで反論を生成し、最後に中立的な要約と主張を強化するための改善案を提示してください。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800,
        )
        # OpenAI の戻り値フォーマットに合わせて抽出
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
        # フォールバック
        return str(response)
    except Exception as e:
        st.error(f"モデル呼び出し中にエラーが発生しました: {e}")
        return ""


# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="AI議論パートナー", page_icon="🤖", layout="centered")
st.title("AI議論パートナー")
st.caption("あなたの主張に対して、反対意見・中立要約・改善案を生成します")

with st.form(key="debate_form"):
    input_mode = st.radio("入力方法を選んでください", ("テキスト入力", "音声アップロード"))

    user_text = ""
    uploaded_audio = None

    if input_mode == "テキスト入力":
        user_text = st.text_area("あなたの主張を入力してください（例：テレワークは効率が悪いと思う）", height=120)
    else:
        uploaded_audio = st.file_uploader("音声ファイルをアップロード（mp3, wav 等）", type=["mp3", "wav", "m4a", "ogg"])
        st.write("※ 音声をアップロードしたら送信ボタンで文字起こし→生成を行います")

    depth = st.selectbox("反論の深さ（目安）", ["短め（要点のみ）", "標準（論点＋具体例）", "詳しく（論理展開＋反証例）"]) 
    tone = st.selectbox("反論のトーン", ["冷静で論理的", "強めで反論的だけど礼儀正しい", "やわらかく説得的"]) 

    submitted = st.form_submit_button("AIに議論してもらう")

if submitted:
    # 入力チェック
    if input_mode == "テキスト入力" and not user_text.strip():
        st.warning("まずはあなたの主張を入力してください。")
    else:
        # 音声がある場合は文字起こし
        if input_mode == "音声アップロード":
            if not uploaded_audio:
                st.warning("音声ファイルをアップロードしてください。")
            else:
                with st.spinner("音声を文字起こししています..."):
                    user_text = transcribe_audio(uploaded_audio)
                    if user_text:
                        st.success("文字起こしが完了しました。以下のテキストを元に議論を生成します。")
                        st.write(user_text)
                    else:
                        st.error("文字起こしに失敗しました。")
        # プロンプト作成
        if user_text and user_text.strip():
            # depth と tone をプロンプトに反映
            depth_map = {
                "短め（要点のみ）": "short",
                "標準（論点＋具体例）": "standard",
                "詳しく（論理展開＋反証例）": "detailed",
            }
            tone_map = {
                "冷静で論理的": "calm and logical",
                "強めで反論的だけど礼儀正しい": "firm but polite",
                "やわらかく説得的": "gentle and persuasive",
            }

            prompt = (
                f"ユーザーの主張: \"{user_text.strip()}\"\n"
                f"タスク: 以下を生成してください。\n"
                f"1) ユーザーの主張に対する反対意見（立場を取って論理的に展開する）。トーン: {tone_map[tone]}。詳細レベル: {depth_map[depth]}。\n"
                f"2) 反対意見の短い中立的な要約（2〜3文）\n"
                f"3) 最後に「あなたの主張をより強くするための改善案」を3つ、実行可能な箇条書きで挙げる。\n"
                f"出力形式: 見出し付き（\"反対意見:\", \"中立要約:\", \"改善案:\"）でわかりやすく。"
            )

            with st.spinner("AIが議論を生成しています..."):
                result = call_model(prompt)

            if result:
                # 表示
                st.subheader("生成結果")
                st.markdown(result)

                # ダウンロード用テキスト
                download_text = f"---\n入力: {user_text.strip()}\n---\n\n{result}"
                st.download_button("結果をテキストでダウンロード", data=download_text, file_name="ai_debate_result.txt")
        else:
            st.warning("処理するテキストがありません。")

# フッター
st.write("---")
st.write("ヒント: トーンや深さを変えることで、さまざまな反論の角度を試せます。OpenAI APIキーは環境変数OPENAI_API_KEY または Streamlit secrets にセットしてください。")
