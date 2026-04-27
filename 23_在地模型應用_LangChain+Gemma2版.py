from langchain_ollama import OllamaLLM

# 使用 Google 開發的輕量化模型 Gemma2 (2b 版本)，適合 CPU 環境運算
llm = OllamaLLM(model = 'gemma2:2b')

# 使用 invoke 函式向模型發送問題 (Prompt)
# 模型會根據其訓練資料，預測並生成關於「四月會發生什麼事」的回答
# 指定回答使用繁體中文，以符合使用者的語言需求
response = llm.invoke("四月會發生甚麼事情？請用繁體中文回答。")

# 輸出模型生成的完整回覆內容
print(response)