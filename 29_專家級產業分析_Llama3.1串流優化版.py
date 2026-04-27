from langchain_ollama import OllamaLLM

# 使用 OllamaLLM 初始化模型，設定為 'llama3.1' (8B) 以獲得更好的繁體中文理解與專業回應
# 將 temperature 設定為 0.7，以平衡創造力與準確性，確保回應既有深度且符合台灣的語言習慣
llm = OllamaLLM(model = 'llama3.1', temperature = 0.7)

# 定義 Prompt：以台灣技術專家角色設定為核心，要求模型進行產業深度分析
# 透過結構化指令（段落條列式）與嚴格的語言規範，確保輸出符合台灣在地專業語感與市場現況
prompt = """
請扮演台灣技術專家，用繁體中文詳細分析台灣目前最熱門的三種程式語言。
要求：
1. 採「段落條列式」呈現，不使用表格。
2. 內容需包含產業應用背景（如：金融、半導體或新創）。
3. 嚴格確保使用台灣慣用語與純繁體中文。
"""

# 使用 stream() 方法進行串流呼叫，即時呈現模型生成的分析內容
for chunk in llm.stream(prompt):

    # end = ""：確保輸出不換行，讓回應能夠連續呈現
    # flush = True：強制刷新輸出緩衝區，使每個 chunk 都能即時顯示，實現流暢的逐字打字機效果
    print(chunk, end = "", flush = True)