import streamlit as st

# 初始化 session_state 儲存各分頁測驗結果
if "results" not in st.session_state:
    st.session_state["results"] = {}  # key: 分頁名稱, value: {題目: 結果}

# 側邊欄：選擇分頁
st.sidebar.title("課程目錄")
page = st.sidebar.selectbox(
    "選擇 stream3 練習",
    ["stream3", "stream3a", "stream3b"]
)

# 函式：存放題目結果到 session_state
def show_result(page_name, question, user_answer, correct_answer):
    if page_name not in st.session_state["results"]:
        st.session_state["results"][page_name] = {}
    if user_answer == "":
        st.session_state["results"][page_name][question] = "無輸入資料"
    elif user_answer == correct_answer:
        st.session_state["results"][page_name][question] = "正確"
    else:
        st.session_state["results"][page_name][question] = "錯誤"

# 函式：顯示該分頁所有結果
def show_results(page_name):
    if page_name in st.session_state["results"]:
        for question, result in st.session_state["results"][page_name].items():
            if result == "正確":
                st.success(f"{question}：{result}")
            elif result == "錯誤":
                st.error(f"{question}：{result}")
            else:
                st.warning(f"{question}：{result}")
    else:
        st.write("尚無測驗結果")

# 分頁 stream3
if page == "stream3":
    st.subheader("文字輸入題")
    answer = st.text_input("請輸入你的答案：")

    if st.button("送出", key = "text_submit"):
        show_result("stream3", "文字輸入題", answer, "正確答案")

    st.subheader("單選題")
    choice = st.radio("哪個是正確答案？", ["A", "B", "C", "D"])

    if st.button("送出", key = "radio_submit"):
        show_result("stream3", "單選題", choice, "A")

# 分頁 stream3a
if page == "stream3a":
    st.title("兩種工具的運用")
    st.subheader("文字輸入題")
    answer = st.text_input("請輸入你的答案：")

    if st.button("送出", key = "text_submit_a"):
        show_result("stream3a", "文字輸入題", answer, "正確答案")

    st.subheader("單選題（有預設選項）")
    choice = st.radio("哪個是正確答案？", ["A", "B", "C", "D"])

    if st.button("送出", key = "radio_submit_a1"):
        show_result("stream3a", "單選題1", choice, "A")

    st.subheader("單選題（無預設選項）")
    choice2 = st.radio("哪個是正確答案？", ["A", "B", "C", "D"], index = None)
    if st.button("送出", key = "radio_submit_a2"):
        show_result("stream3a", "單選題2", choice2, "B")

# 分頁 stream3b
if page == "stream3b":
    st.title("兩種工具的運用")
    st.subheader("文字輸入題")
    answer = st.text_input("請輸入你的答案：")

    if st.button("送出", key = "text_submit_b"):
        show_result("stream3b", "文字輸入題", answer, "正確答案")

    st.subheader("單選題（有預設選項）")
    choice = st.radio("哪個是正確答案？", ["A", "B", "C", "D"])

    if st.button("送出", key = "radio_submit_b1"):
        show_result("stream3b", "單選題1", choice, "A")

    st.subheader("單選題（無預設選項）")
    choice2 = st.radio("哪個是正確答案？", ["A", "B", "C", "D"], index = None)

    if st.button("送出", key = "radio_submit_b2"):
        show_result("stream3b", "單選題2", choice2, "B")

    st.subheader("單選題（水平排列）")
    choice3 = st.radio("哪個是正確答案？", ["A", "B", "C", "D"], index = None, horizontal = True)
    if st.button("送出", key = "radio_submit_b3"):
        show_result("stream3b", "單選題3", choice3, "C")

# 分頁底部顯示結果
st.divider()
st.subheader("測驗結果：")
show_results(page)