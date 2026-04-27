import asyncio
from langchain_ollama import OllamaLLM

# 使用 OllamaLLM 初始化模型，設定為 'imac/llama3.2-taiwan:3b-instruct-q8_0' (3B 模型)
# 這是一個針對台灣繁體中文優化的輕量模型，速度快且回應在地化
# 將 temperature 設定為 0.2，以獲得更穩定和準確的回應
llm = OllamaLLM(model = "imac/llama3.2-taiwan:3b-instruct-q8_0", temperature = 0.2)

# 定義一個非同步函式 async_batch()，用於執行批次模型請求
async def async_batch():
    print(">>> 正在發送 abatch() 非同步批量請求...")

    # 使用 await llm.abatch() 進行非同步批量請求
    # 相比於同步的 batch()，abatch 能讓程式在等待模型回傳的過程中，不阻塞（鎖死）其他任務的執行
    responses = await llm.abatch([
        "台灣的熱門程式語言有哪些？",
        "台灣的 AI 產業發展如何？",
        "台灣的開發者社群有哪些？"
    ])

    # 使用 enumerate 取得索引 (i) 與對應的回答內容 (res)
    for i, res in enumerate(responses):

        # 印出問題編號（i + 1 是因為索引從 0 開始）以及模型生成的詳細回答
        # 在每個回答後加上 \n 換行符號，讓輸出更清晰易讀
        print(f"問題 {i + 1}：{res}\n")

# 透過 asyncio.run() 啟動事件迴圈並執行非同步函式
if __name__ == "__main__":
    asyncio.run(async_batch())