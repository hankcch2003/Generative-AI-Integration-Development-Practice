from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# 初始化 Ollama LLM（使用 llama3.2:latest）
llm = OllamaLLM(model = "llama3.2:latest")

# Prompt 模板（強制繁體中文 + 一句話）
prompt = PromptTemplate.from_template(
    "你是一個專業且嚴格的繁體中文助手。\n"
    "請只用一句話回答，內容必須全程使用繁體中文，禁止英文或混合語言。\n"
    "請介紹：{topic}"
)

# 前處理：轉換輸入格式
preprocess = RunnableLambda(lambda x: {"topic": x})

# 後處理：清理引號 + 格式化輸出
postprocess = RunnableLambda(
    lambda x: "[結果] " + str(x).replace("“", "").replace("”", "").replace('"', "")
)

# LCEL pipeline
# preprocess → prompt → llm → postprocess
chain = preprocess | prompt | llm | postprocess

# 批次輸入資料
topics = ["貓", "狗", "兔子"]

# batch 執行
results = chain.batch(topics)

# 輸出模型結果
print("=== 模型結果 ===")
print()

for r in results:
    print(r)