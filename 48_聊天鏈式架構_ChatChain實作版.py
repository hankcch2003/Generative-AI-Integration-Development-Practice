from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 初始化 OllamaLLM 物件，並指定模型名稱為 'llama3.2:latest'
llm = OllamaLLM(model = 'llama3.2:latest')

# 建立聊天提示模板，包含系統角色與使用者輸入（帶有 subject 變數）
chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一個智慧且友善的助手"),
    ("user", "請講一個與 {subject} 有關的笑話")
])

# LCEL 串接流程：Prompt → LLM → 字串輸出解析
chain = chat_template | llm | StrOutputParser()

# 執行 chain（subject = 狗狗）
response = chain.invoke({"subject": "狗狗"})
print('模型輸出：')
print(response)

# 執行 chain（subject = 貓貓）
response = chain.invoke({"subject": "貓貓"})
print('模型輸出：')
print(response)