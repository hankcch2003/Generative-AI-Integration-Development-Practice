from langchain_ollama import OllamaLLM

# 使用 Google 開發的輕量化模型 Gemma2 (2b 版本)，適合 CPU 環境運算
llm = OllamaLLM(model = 'gemma2:2b')

# 角色演繹測試：專業氣象預報員
# 測試模型是否能切換至「專業、數據導向」的語氣，並提供功能性建議
prompt = """
請扮演一位台灣中央氣象署的預報員，用「專業且嚴謹」的口吻，
向民眾說明台灣四月的天氣特徵，包含氣溫趨勢、降雨機率與穿衣建議。
請用繁體中文回答。
"""

# 使用 invoke 函式將結構化的提示詞發送給 LLM 並獲取生成結果
response = llm.invoke(prompt)

# 輸出模型生成的完整回覆內容
print(response)