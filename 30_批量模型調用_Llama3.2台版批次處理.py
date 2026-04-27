from langchain_ollama import OllamaLLM

# 使用 OllamaLLM 初始化模型，設定為 'imac/llama3.2-taiwan:3b-instruct-q8_0' (3B 模型)
# 這是一個針對台灣繁體中文優化的輕量模型，速度快且回應在地化
# 將 temperature 設定為 0.7，以平衡創造力與準確性，確保回應既有深度且符合台灣的語言習慣
llm = OllamaLLM(model = "imac/llama3.2-taiwan:3b-instruct-q8_0", temperature = 0.7)

# 定義一個字串清單，包含所有想要詢問模型的問題（Prompt）
prompts = [
    "請介紹台灣的熱門程式語言。",
    "機器學習和深度學習有什麼區別？",
    "人工智慧的應用有哪些？"
]

# 調用 llm 的 batch 方法，一次性將整個問題清單發送給 Ollama 模型
# 這會回傳一個包含所有回答的清單 (list)，順序與 prompts 一一對應
responses = llm.batch(prompts)

# 使用 enumerate 取得索引 (i) 與對應的回答內容 (res)
for i, res in enumerate(responses):

    # 印出問題編號（i + 1 是因為索引從 0 開始）以及模型生成的詳細回答
    # 在每個回答後加上 \n 換行符號，讓輸出更清晰易讀
    print(f"問題 {i + 1}：{res}\n")