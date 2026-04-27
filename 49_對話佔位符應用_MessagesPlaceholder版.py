from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

# 建立聊天提示模板（system + 動態 user messages 區塊）
chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一個善於解答問題的助手"),
    MessagesPlaceholder("user_msgs")
])

# 模擬多輪使用者訊息輸入
user_messages = [
    HumanMessage(content = "你好！"),
    HumanMessage(content = "請問 Python 和 Java 有什麼不同？"),
    HumanMessage(content = "如果要學 AI，推薦什麼語言？"),
    HumanMessage(content = "感謝你的幫忙！")
]

# 將多則 user messages 帶入模板並產生 messages 結果
response = chat_template.invoke({
    "user_msgs": user_messages
})

# 輸出 ChatPromptTemplate 結果
print('模型輸出：')
print(response)