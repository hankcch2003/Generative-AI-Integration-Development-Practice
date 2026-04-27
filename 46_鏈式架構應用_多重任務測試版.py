from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

# 初始化 OllamaLLM 物件，並指定模型名稱為 'phi3:mini'
llm = OllamaLLM(model = 'phi3:mini')

# 定義輸入樣板，設定模型的人設為「善於解答問題的助手」
template = PromptTemplate.from_template("你是一個善於解答問題的助手：{subject}")

# 建立處理鏈：樣板 | 模型 | 輸出解析器
# 透過管道符號 | 將步驟串聯，StrOutputParser 會將結果直接轉為純字串輸出
chain = template | llm | StrOutputParser()

# 執行第一項任務：基本問候測試
# 觀察模型是否能以助手的身份進行親切回應
response = chain.invoke({"subject": "你好！"})
print('\n[任務 1] 模型輸出：')
print(response)

# 執行第二項任務：程式語言知識對比
# 觀察 Phi-3 對於 Java 與 Python 差異的邏輯分點能力
response = chain.invoke({"subject": "請問 Python 和 Java 有什麼不同？"})
print('\n[任務 2] 模型輸出：')
print(response)

# 執行第三項任務：學習路徑與專業建議
# 測試模型在 AI 領域的推薦邏輯
response = chain.invoke({"subject": "如果要學 AI，推薦什麼語言？"})
print('\n[任務 3] 模型輸出：')
print(response)

# 執行第四項任務：結束語與禮貌回應
# 測試模型是否能維持助手人設完成對話閉環
response = chain.invoke({"subject": "感謝你的幫忙！"})
print('\n[任務 4] 模型輸出：')
print(response)