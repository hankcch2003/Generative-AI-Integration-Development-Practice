from langchain_ollama import OllamaLLM

# 使用 OllamaLLM 初始化模型，設定為 'llama3.1' (8B) 以獲得更好的繁體中文理解與專業回應
# 同時將 temperature 設定為 0.7，以平衡創造力與準確性，確保回應既有深度又符合台灣的語言習慣
llm = OllamaLLM(model = 'llama3.1', temperature = 0.7)

# 使用 stream() 方法進行串流呼叫，即時呈現模型生成的分析內容
for chunk in llm.stream("台灣的熱門程式語言有哪些？"):

    # end = ""：確保輸出不換行，讓回應能夠連續呈現
    # flush = True：強制刷新輸出緩衝區，使每個 chunk 都能即時顯示，實現流暢的逐字打字機效果
    print(chunk, end = "", flush = True)