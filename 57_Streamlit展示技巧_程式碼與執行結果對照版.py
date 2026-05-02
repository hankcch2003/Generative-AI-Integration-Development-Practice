import streamlit as st

# 程式碼區塊（不執行）
st.code("""
for i in range(5):
    print(i)
""", language = "python")

st.divider()

# 說明文字
st.write("執行結果：")

# 實際執行結果
for i in range(5):
    st.write(i)