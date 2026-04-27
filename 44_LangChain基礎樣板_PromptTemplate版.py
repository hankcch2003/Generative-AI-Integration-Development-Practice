from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# 初始化 OllamaLLM 物件，並指定模型名稱為 'gemma2:2b'
llm = OllamaLLM(model = 'gemma2:2b')

# 定義輸入樣板，使用 {subject} 作為可替換的主題變數
template = PromptTemplate.from_template("請分享一個與{subject}有關的笑話")

# 執行樣板組合，將主題變數設定為 "狗狗"
response = template.invoke({"subject": "狗狗"})

# 輸出樣板處理後產生的完整問題內容
print('輸入樣板的輸出：')
print(response)

# 將組合好的問題傳送給模型執行推論
response = llm.invoke(response)

# 輸出模型最終回傳的回答內容
print('\n模型輸出：')
print(response)