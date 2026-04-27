from transformers import pipeline

# 指定使用 MoritzLaurer 的多語系模型進行 Zero-shot 文本分類
model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# 建立 zero-shot-classification 的 pipeline，並載入預訓練模型
classifier = pipeline("zero-shot-classification", model = model_name)

# 定義要分析的文本內容
text = "志玲姊姊穿著日本火腿隊球衣，緊身牛仔褲凸顯了她的修長美腿。"

# 定義自定義候選標籤
candidate_labels = ['體育', '娛樂', '國際']

# 傳入待分析文本並指定候選標籤，並使用自定義的假設模板來進行分類
result = classifier(text, candidate_labels, hypothesis_template = "這篇文章的類別是{}。")

# 輸出分析文本與預測結果
print(f"文字: {text}")

# 輸出最高機率類別（模型會自動將結果依機率降序排列）
# 取得 labels 清單中的第一個元素及其分數，並格式化為小數點後兩位的百分比形式
print(f"最高機率類別: {result['labels'][0]} ({result['scores'][0]:.2%})")