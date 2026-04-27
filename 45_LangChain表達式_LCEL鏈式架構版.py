from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

# 初始化 OllamaLLM 物件，並指定模型名稱為 'llama3.2:latest'
llm = OllamaLLM(model = 'llama3.2:latest')

# 定義輸入樣板，使用 {subject} 作為可替換的主題變數
template = PromptTemplate.from_template("請分享一個與{subject}有關的笑話")

# 建立處理鏈：樣板 | 模型 | 輸出解析器
# 透過管道符號 | 將步驟串聯，StrOutputParser 會將結果直接轉為純字串
chain = template | llm | StrOutputParser()

# 執行處理鏈並傳入參數，將主題變數設定為 "狗狗"
response = chain.invoke({"subject": "狗狗"})

# 輸出模型最終回傳的回答內容
print('模型輸出：')
print(response)