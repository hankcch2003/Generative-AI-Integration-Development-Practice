from transformers import pipeline

# 指定使用針對中文優化過的「二郎神」系列模型 (Roberta-110M) 來進行情緒分析
classifier = pipeline('sentiment-analysis', model = 'IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment')

# 將想要分析的中文句子放入清單 (List) 中傳給 classifier
# 模型會分析每一句話的情緒傾向（正面或負面）以及信心分數
result = classifier(["我非常喜歡", "我非常討厭"])

# 輸出模型的分析結果，包含每句話的情緒標籤和對應的信心分數
# 結果會以列表的形式呈現，每個元素包含情緒類別 (label) 與信心分數 (score)
# 例如：Positive 代表正面情緒，Negative 代表負面情緒，score 則代表模型判斷的信心程度
print(result)