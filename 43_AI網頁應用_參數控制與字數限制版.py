import streamlit as st
from langchain_ollama import OllamaLLM
import io

# 設定網頁的主標題，會以最大的字體顯示在頁面頂端
st.title("多模型 LLM 平台")

# 定義一個清單，包含所有可以在介面中選擇的 Ollama 模型名稱
models = [
    "gemma2:2b",
    "llama3.2:latest",
    "phi3:mini",
    "qwen3-vl:4b",
    "only test"
]

# 建立多選下拉選單，讓使用者選擇模型，預設會選取清單中所有的模型
selected_models = st.multiselect("選擇模型：", models, default = models)

# 在網頁上建立三個文字輸入框，供使用者輸入問題
q1 = st.text_input("問題 1")
q2 = st.text_input("問題 2")
q3 = st.text_input("問題 3")

# 使用列表推導式重新處理：過濾掉只包含空白的輸入，確保只有真正有文字的問題才會被送去模型執行
prompts = [p for p in [q1, q2, q3] if p.strip()] 

# 建立「執行」按鈕，當點擊時執行下方的運算邏輯
if st.button("執行"):

    # 如果使用者沒有輸入任何有效問題，顯示警告訊息
    if not prompts:
        st.warning('請至少輸入一個問題')
    else:
        # 使用迴圈逐一處理使用者所選取的每一個模型名稱
        for model_name in selected_models:

            # 初始化記憶體內的字串緩衝區物件，用來暫存所有的問答內容
            buffer = io.StringIO() 

            # 在網頁上顯示目前正在回應的模型名稱作為副標題
            st.subheader(f"{model_name}")
            
            # 使用 try...except 捕捉異常，確保單一模型出錯時不會中斷整個程式
            try:
                # 根據當前的模型名稱初始化 OllamaLLM 物件，並設定生成參數
                llm = OllamaLLM(
                    model = model_name,
                    temperature = 0.3, # 設定隨機性為 0.3：數值越小回答越固定，增加結果穩定性
                    num_predict = 200  # 設定長度限制：限制最多生成 200 個字元，避免資源過度消耗
                )

                # 在文字緩衝區內寫入目前模型的分隔線
                buffer.write(f"--- 模型：{model_name} ---\n")
                
                # 遍歷問題列表，用 invoke 逐一呼叫模型處理，避免 batch 同時執行導致記憶體溢位
                for i, p in enumerate(prompts):

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
                    label = "下載結果 (.txt)",         # 設定按鈕上顯示的文字名稱
                    data = buffer.getvalue(),         # 從記憶體緩衝區提取出所有累積的文字數據 
                    file_name = f"{model_name}.txt",  # 設定下載檔案名稱（以當前模型的名稱命名）
                    mime = "text/plain"               # 指定檔案格式為純文字
                )
                
            # 若該模型執行過程中出錯，捕捉並顯示紅色的錯誤提示訊息，方便偵錯
            except Exception as e:
                st.error(f'模型 {model_name} 執行出錯')

                # 顯示具體的錯誤原因，避免程式黑畫面當機
                st.info(f'錯誤訊息：{str(e)}')