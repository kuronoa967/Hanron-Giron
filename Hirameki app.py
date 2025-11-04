import streamlit as st
import openai

st.title("💡 AIひらめきノート")
st.write("思いついた言葉を入力して、AIに関連アイデアを考えてもらいましょう。")

keyword = st.text_input("ひらめきの種を入力してください:")

if st.button("アイデアを生成"):
    if keyword:
        with st.spinner("AIが考え中です..."):
            # 仮のAI出力（後でAPI接続に変更）
            ideas = [f"{keyword} × 未来", f"{keyword} × 日常", f"{keyword} × テクノロジー"]
            st.success("AIの提案:")
            for idea in ideas:
                st.markdown(f"- {idea}")
    else:
        st.warning("キーワードを入力してください。")
