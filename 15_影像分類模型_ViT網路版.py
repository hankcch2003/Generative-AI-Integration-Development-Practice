from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from PIL import Image
import requests

# 設定模型 ID，這裡使用的是 Google 的 ViT 基礎影像分類模型
model_name = 'google/vit-base-patch16-224'

# 載入處理器（負責調整圖片大小與標準化）與影像分類模型本體
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# 指定圖片網址並使用 requests 取得原始影像內容
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream = True).raw)

# 將圖片處理成張量（Tensors）並設為 PyTorch 格式（pt）
inputs = processor(images = image, return_tensors = "pt")

# 關閉梯度計算（節省記憶體與運算資源），因為現在只是推論而非訓練
with torch.no_grad():
    # 取得模型輸出結果
    outputs = model(**inputs)

    # 提取影像分類的原始分數（Logits）
    # ViT 模型使用 .logits 而非 .logits_per_image
    logits = outputs.logits 

# 找出 Logits 分數最高（信心度最高）的類別索引
predicted_class_idx = logits.argmax(-1).item()

# 將預測結果的索引透過模型配置中的 id2label 轉換為文字標籤
print("預測結果:", model.config.id2label[predicted_class_idx])