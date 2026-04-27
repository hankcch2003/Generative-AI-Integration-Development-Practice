from langchain_ollama import OllamaLLM

# 使用 Google 開發的輕量化模型 Gemma2 (2b 版本)，適合 CPU 環境運算
llm = OllamaLLM(model = 'gemma2:2b')

# 使用簡潔的指令測試模型的摘要能力
# 要求模型以「繁體中文」並透過「條列式」輸出，適合快速獲取重點
prompt = "請用繁體中文、條列式說明台灣四月的三個核心關鍵字。"

# 使用 invoke 函式將結構化的提示詞發送給 LLM 並獲取生成結果
response = llm.invoke(prompt)

# 輸出模型生成的完整回覆內容
print(response)