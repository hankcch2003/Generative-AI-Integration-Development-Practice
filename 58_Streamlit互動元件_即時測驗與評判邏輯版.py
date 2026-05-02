import streamlit as st

# 初始化 session_state
if "text_result" not in st.session_state:
    st.session_state.text_result = ""

if "mcq_result" not in st.session_state:
    st.session_state.mcq_result = ""

# 輸入區塊
answer = st.text_input("請輸入你的答案：")

st.divider()

# 按鈕判斷（輸入題）
if st.button("送出"):

    if answer == "正確答案":
        st.session_state.text_result = "答對了！"
    else:
        st.session_state.text_result = "再想想看"

# 永久顯示結果
if st.session_state.text_result:
    if st.session_state.text_result == "答對了！":
        st.success(st.session_state.text_result)
    else:
        st.error(st.session_state.text_result)

st.divider()

# 單選題
choice = st.radio("哪個是正確答案？", ["A", "B", "C", "D"])
correct = "B"

# 按鈕判斷（單選題）
if st.button("提交選擇題"):

    if choice == correct:
        st.session_state.mcq_result = "答對了"
    else:
        st.session_state.mcq_result = "答錯了"

# 永久顯示結果
if st.session_state.mcq_result:
    if st.session_state.mcq_result == "答對了":
        st.success(st.session_state.mcq_result)
    else:
        st.error(st.session_state.mcq_result)