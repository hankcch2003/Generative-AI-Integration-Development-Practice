from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 初始化 Ollama LLM（使用 llama3.2:latest）
llm = OllamaLLM(model = "llama3.2:latest")

# 建立 Prompt（system / history / input）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個專業且穩定的繁體中文助手。回答要清楚一致、條理分明，不要中英混用。"),
    MessagesPlaceholder(variable_name = "chat_history"),
    ("human", "{input}")
])

# LCEL：Prompt → LLM
chain = prompt | llm

# 對話記憶（手動管理）
chat_history = []

# 第一次輸入
payload = {
    "input": "你好，請用 30 字介紹你自己。",
    "chat_history": chat_history
}

print("=== invoke 模式（第一次） ===")
print()
response = chain.invoke(payload)
print(response)

# 更新記憶
chat_history.append(HumanMessage(content = payload["input"]))
chat_history.append(AIMessage(content = response))

# 第二次輸入
payload = {
    "input": "剛剛你說了什麼？",
    "chat_history": chat_history
}

print()
print("=== invoke 模式（第二次，使用記憶） ===")
print()
response = chain.invoke(payload)
print(response)

# 更新記憶
chat_history.append(HumanMessage(content = payload["input"]))
chat_history.append(AIMessage(content = response))

# stream 測試
payload = {
    "input": "請再用更白話解釋一次",
    "chat_history": chat_history
}

print()
print("=== stream 模式（使用記憶） ===")
print()

for chunk in chain.stream(payload):
    print(chunk, end = "", flush = True)

print()
print("=== 測試結束 ===")