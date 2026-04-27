from langchain_ollama import OllamaLLM

# 使用 OllamaLLM 初始化模型，設定為 'gemma2:2b' (2B 模型)
# 選擇更輕量的模型以降低 CPU 負載，確保在大負荷下能穩定完成檔案寫入任務
# 將 temperature 設定為 0.2，以獲得更穩定和準確的回應
llm = OllamaLLM(model = "gemma2:2b", temperature = 0.2)

# 修改後的指令清單：加入「字數限制」與「簡短格式」
# 這樣做能減少模型生成的內容長度，有效提升 CPU 運算的反應速度，避免程式卡死
prompts = [
    "請用 50 字以內，簡短介紹台灣目前最熱門的 3 種程式語言。",
    "請用一句話說明機器學習和深度學習的最主要區別。",
    "請列出 3 個目前最常見的人工智慧應用，不用詳細解釋。"
]

# 設定輸出的檔案名稱為 '33_Gemma2執行結果_file1_gemma2.txt'，用於與 llama 3.2 的執行效能進行對照分析
file1 = '33_Gemma2執行結果_file1_gemma2.txt'

# 調用 llm 的 batch 方法，一次性將整個問題清單發送給 Ollama 模型
# 這會回傳一個包含所有回答的清單 (list)，順序與 prompts 一一對應
responses = llm.batch(prompts)

# 使用 'w' (write) 模式開啟檔案，若檔案已存在會先清空內容再寫入
# encoding = 'utf-8' 確保繁體中文寫入時不會出現亂碼
with open(file1, 'w', encoding = 'utf-8') as f:

    # 使用 enumerate 取得索引 (i) 與對應的回答內容 (res)
    for i, res in enumerate(responses):

        # 印出問題編號（i + 1 是因為索引從 0 開始）以及模型生成的詳細回答
        # 在每個回答後加上 \n 換行符號，讓輸出更清晰易讀
        output = f"問題 {i + 1}：{res}\n"

        # 正式將格式化後的內容寫入到文字檔中
        f.write(output)