import streamlit as st
from langchain_ollama import OllamaLLM
import io

# 設定網頁的主標題，這會以最大的字體顯示在頁面頂端
st.title("多模型 LLM 平台")

# 定義一個清單，包含所有可以在介面中選擇的 Ollama 模型名稱
models = [
    "gemma2:2b",
    "llama3.2:latest",
    "phi3:mini",
    "qwen3-vl:4b"
]

# 建立一個多選下拉選單，讓使用者挑選模型，預設會選取清單中所有的模型
selected_models = st.multiselect("選擇模型：", models, default = models)

# 在網頁上建立三個單行文字輸入框，讓使用者分別輸入三個想要詢問的問題
q1 = st.text_input("問題 1")
q2 = st.text_input("問題 2")
q3 = st.text_input("問題 3")

# 初始化一個記憶體內的字串緩衝區物件，用來暫存所有的問答內容
buffer = io.StringIO()

# 建立一個「執行」按鈕，當使用者用滑鼠點擊時，會執行縮排內的程式碼邏輯
if st.button("執行"):

    # 將使用者輸入的內容彙整成列表
    prompts = [q1, q2, q3]
    
    # 使用迴圈逐一處理使用者所選取的每一個模型名稱
    for model_name in selected_models:

        # 在網頁上顯示目前正在回應的模型名稱作為副標題
        st.subheader(f"模型：{model_name}")
        
        # 根據目前的模型名稱，初始化 OllamaLLM 物件
        llm = OllamaLLM(model = model_name)
        
        # 在文字緩衝區內寫入目前模型的分隔線
        buffer.write(f"--- 模型：{model_name} ---\n")
        
        # 遍歷問題列表，用 invoke 逐一呼叫模型處理，避免 batch 同時執行導致記憶體溢位
        for i, p in enumerate(prompts):

            # 只有當問題輸入框不為空時才執行，以節省運算資源
            if p.strip():

                # 執行單次模型呼叫 (Invoke)，讓回答逐一產出
                res = llm.invoke(p)

                # 在網頁畫面上顯示問題編號及內容
                st.write(f"**問題 {i + 1}：** {p}")

                # 在網頁畫面上顯示模型產出的回答內容
                st.write(f"{res}")
                
                # 將格式化後的問答內容寫入記憶體緩衝區
                buffer.write(f"問題 {i + 1}：{p}\n回答：{res}\n\n")
        
    # 當所有選取的模型都跑完後，在頁面底部顯示一個下載按鈕
    st.download_button(
        label = "下載結果 (.txt)",                    # 設定按鈕上顯示的文字名稱
        data = buffer.getvalue(),                    # 從記憶體緩衝區提取出所有累積的文字數據
        file_name = "40_網頁執行結果_gemma2_2b.txt",  # 設定下載檔案的名稱（預設下載檔名：file_name = "40_網頁執行結果_gemma2_2b.txt"）
        mime = "text/plain"                          # 指定檔案格式為純文字
    )