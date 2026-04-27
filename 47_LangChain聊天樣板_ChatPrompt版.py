from langchain_core.prompts import ChatPromptTemplate

# 建立聊天提示模板，包含系統角色與使用者輸入（帶有 subject 變數）
chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一個智慧且友善的助手"),
    ("user", "請講一個與 {subject} 有關的笑話")
])

# 將 subject 帶入模板，產生格式化後的 messages
response = chat_template.invoke({"subject": "狗狗"})

# 輸出 ChatPromptTemplate 產生的 messages 結果
print('輸入樣板的輸出：')
print(response)