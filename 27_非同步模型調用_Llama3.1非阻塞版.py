import asyncio
from langchain_ollama import OllamaLLM

# 使用 OllamaLLM 初始化模型，設定為 'llama3.1' (8B) 以獲得更好的繁體中文理解與專業回應
# 同時將 temperature 設定為 0.7，以平衡創造力與準確性，確保回應既有深度又符合台灣的語言習慣
llm = OllamaLLM(model = 'llama3.1', temperature = 0.7)

# 定義一個非同步函式 async_request()，用於執行模型請求
async def async_request():
    print(">>> 正在發送 ainvoke() 非同步請求...")
    
    # 使用 await 等待非同步回應，這能讓程式在等待時不被鎖死
    response = await llm.ainvoke("台灣的熱門程式語言有哪些？")
    
    # 輸出模型生成的完整回覆內容
    print(response)

# 透過 asyncio.run() 啟動事件迴圈並執行非同步函式
if __name__ == "__main__":
    asyncio.run(async_request())