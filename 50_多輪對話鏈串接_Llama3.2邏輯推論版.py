from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

# 初始化 OllamaLLM 物件，並指定模型名稱為 'llama3.2:latest'
llm = OllamaLLM(model = 'llama3.2:latest')

# 多輪對話提示模板
chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一個善於解答問題的助手"),
    MessagesPlaceholder("user_msgs")
])

# 模擬多輪使用者輸入
user_messages = [
    HumanMessage(content = "你好！"),
    HumanMessage(content = "請問 Python 和 Java 有什麼不同？"),
    HumanMessage(content = "如果要學 AI，推薦什麼語言？"),
    HumanMessage(content = "感謝你的幫忙！")
]

# LCEL流程：Prompt → LLM → Output Parser
chain = chat_template | llm | StrOutputParser()

# 執行 chain 並輸入 messages
response = chain.invoke({"user_msgs": user_messages})

# 輸出 ChatPromptTemplate 結果
print('模型輸出：')
print(response)