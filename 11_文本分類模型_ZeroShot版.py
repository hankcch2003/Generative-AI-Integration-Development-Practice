from transformers import pipeline

# 指定使用 zero-shot-classification 模型來進行文本分類
# 這個模型可以在沒有特定訓練的情況下，根據文本內容和候選標籤來進行分類
classifier = pipeline('zero-shot-classification')

# 定義要分析的文本內容
text = "志玲姊姊穿著日本火腿隊球衣，緊身牛仔褲凸顯了她的修長美腿。"

# 傳入待分析文本並指定「自定義」候選標籤
# 模型會根據語義關聯性，計算每個標籤的可能性百分比
result = classifier(text, candidate_labels = ['體育', '娛樂', '國際'])

# 輸出模型的分類結果 (以字典形式呈現)，包含原始文本 (sequence)、類別標籤 (labels) 與機率分數 (scores)
# 數值最高的標籤表示模型認為該文本最有可能屬於的類別
print(result)