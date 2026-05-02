import streamlit as st

# 初始化結果狀態
if "text_result" not in st.session_state:
    st.session_state.text_result = ""

if "result1" not in st.session_state:
    st.session_state.result1 = ""

if "result2" not in st.session_state:
    st.session_state.result2 = ""

# 標題區塊
st.title("兩種工具的運用")

# 輸入題區塊
st.subheader("輸入文字，可嘗試輸入正確答案")
answer = st.text_input("請輸入你的答案：")

# 按鈕判斷
if st.button("送出"):
    if answer == "正確答案":
        st.session_state.text_result = "答對了！"
    elif answer == "":
        st.session_state.text_result = "沒有輸入資料！"
    else:
        st.session_state.text_result = "再想想看！"

# 顯示結果
if st.session_state.text_result == "答對了！":
    st.success(st.session_state.text_result)
elif st.session_state.text_result == "沒有輸入資料！":
    st.warning(st.session_state.text_result)
elif st.session_state.text_result == "再想想看！":
    st.error(st.session_state.text_result)

st.divider()

# 單選題（有預設）
st.subheader("選擇鈕，可選擇一個")
choice = st.radio("哪個是正確答案？", ["A", "B", "C", "D"], key = "q1")
correct = "A"

if st.button("送出選擇題（有預設）"):
    if choice == correct:
        st.session_state.result1 = "答對了！"
    else:
        st.session_state.result1 = "答錯了！"

# 顯示結果（永遠存在）
if st.session_state.result1:
    if st.session_state.result1 == "答對了！":
        st.success(st.session_state.result1)
    else:
        st.error(st.session_state.result1)

st.divider()

# 單選題（無預設）
st.subheader("沒有預設選項")
choice2 = st.radio("哪個是正確答案？", ["A", "B", "C", "D"], index = None, key = "q2")

if st.button("送出選擇題（無預設）"):
    if choice2 is None:
        st.session_state.result2 = "請先選擇答案！"
    elif choice2 == correct:
        st.session_state.result2 = "答對了！"
    else:
        st.session_state.result2 = "答錯了！"

# 顯示結果（永遠存在）
if st.session_state.result2:
    if st.session_state.result2 == "請先選擇答案！":
        st.warning(st.session_state.result2)
    elif st.session_state.result2 == "答對了！":
        st.success(st.session_state.result2)
    else:
        st.error(st.session_state.result2)