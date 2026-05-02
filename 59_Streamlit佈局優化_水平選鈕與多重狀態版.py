import streamlit as st

# 初始化 session_state
if "text_result" not in st.session_state:
    st.session_state.text_result = ""

if "radio1_result" not in st.session_state:
    st.session_state.radio1_result = ""

if "radio2_result" not in st.session_state:
    st.session_state.radio2_result = ""

if "radio3_result" not in st.session_state:
    st.session_state.radio3_result = ""

# 標題
st.title("兩種工具的運用")

# 輸入題
st.subheader("輸入文字，可嘗試輸入正確答案")
answer = st.text_input("請輸入你的答案：")

if st.button("送出"):
    if answer == "正確答案":
        st.session_state.text_result = "答對了！"
    elif answer == "":
        st.session_state.text_result = "沒有輸入資料！"
    else:
        st.session_state.text_result = "再想想看！"

# 顯示結果（永遠存在）
if st.session_state.text_result:
    if st.session_state.text_result == "答對了！":
        st.success(st.session_state.text_result)
    elif st.session_state.text_result == "沒有輸入資料！":
        st.warning(st.session_state.text_result)
    else:
        st.error(st.session_state.text_result)

st.divider()

# 單選題（有預設）
st.subheader("選擇鈕，可選擇一個")
choice = st.radio("哪個是正確答案？", ["A", "B", "C", "D"], key = "r1")

if st.button("送出選擇題（有預設）"):
    st.session_state.radio1_result = choice

# 顯示結果（永遠存在）
if st.session_state.radio1_result:
    st.success(st.session_state.radio1_result)

st.divider()

# 單選題（無預設）
st.subheader("沒有預設選項")
choice2 = st.radio("哪個是正確答案？", ["A", "B", "C", "D"], index = None, key = "r2")

if st.button("送出選擇題（無預設）"):
    if choice2 is None:
        st.session_state.radio2_result = "請先選擇答案！"
    else:
        st.session_state.radio2_result = choice2

# 顯示結果（永遠存在）
if st.session_state.radio2_result:
    if st.session_state.radio2_result == "請先選擇答案！":
        st.warning(st.session_state.radio2_result)
    else:
        st.success(st.session_state.radio2_result)

st.divider()

# 單選題（水平排列）
st.subheader("沒有預設選項且水平排列")
choice3 = st.radio("哪個是正確答案？", ["A", "B", "C", "D"], index = None, horizontal = True, key = "r3")

if st.button("送出選擇題（水平）"):
    if choice3 is None:
        st.session_state.radio3_result = "請先選擇答案！"
    else:
        st.session_state.radio3_result = choice3

# 顯示結果（永遠存在）
if "radio3_result" in st.session_state:
    if st.session_state.radio3_result == "請先選擇答案！":
        st.warning(st.session_state.radio3_result)
    else:
        st.success(st.session_state.radio3_result)