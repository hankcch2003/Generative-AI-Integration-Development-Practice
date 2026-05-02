import asyncio
from datetime import datetime

# 取得目前時間（時:分:秒）
def get_now():
    return datetime.now().strftime("%H:%M:%S")

# async + return（一次回傳全部結果）
async def example1(n):
    print(f"[{get_now()}] 準備進行")
    result = []

    for i in range(n):
        await asyncio.sleep(1.5)
        result.append(f"鈔票{i + 1}")

    print(f"[{get_now()}] 準備輸出")
    return result

# async + yield（逐筆串流回傳）
async def example2(n):
    for i in range(n):
        print(f"[{get_now()}] 準備進行")
        await asyncio.sleep(1.5)
        print(f"[{get_now()}] 準備輸出")
        yield f"鈔票{i + 1}"

# 主流程（執行 async 任務）
async def main():
    print("=== return 練習 ===")
    print()
    money1 = await example1(3)
    for cash in money1:
        print(f"收到 {cash}")

    print()
    print("=== yield 練習 ===")
    print()
    async for cash in example2(3):
        print(f"收到 {cash}")

# 啟動事件迴圈
asyncio.run(main())