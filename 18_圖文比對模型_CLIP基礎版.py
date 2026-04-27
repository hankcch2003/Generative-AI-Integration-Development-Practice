import warnings
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# 忽略警告訊息（例如版本相容性警告），讓輸出畫面保持乾淨
warnings.filterwarnings("ignore")

# 載入處理器（負責圖片縮放、標準化與文字 Tokenize）與 CLIP 模型
# 這裡使用的是 OpenAI 的 CLIP 模型（ViT-B/32），常用於圖文比對、影像檢索或零樣本分類
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# 指定圖片的絕對路徑（使用 r 避免反斜線被誤認為轉義字元）
image_path = r"E:\Python\Python4-人工智慧整合開發實務\20260409\test.png"

# 使用 PIL 讀取圖片，並轉換為 RGB 標準格式（避免灰階或透明圖層造成錯誤）
raw_image = Image.open(image_path).convert('RGB')

# 定義候選標籤，模型會計算圖片與這些文字描述的相似度
labels = ["a dog", "a cat", "a person", "a car", "a sofa"]

# 將文字與圖片處理成張量（Tensors）並設為 PyTorch 格式（pt）
inputs = processor(text = labels, images = raw_image, return_tensors = "pt", padding = True)

# 取得模型輸出的原始相似度分數（Logits）
outputs = model(**inputs)

# 從輸出中提取圖片對文字的原始分數（Logits）
logits_per_image = outputs.logits_per_image 

# 使用 Softmax 將原始分數轉換為總和為 1 的機率分佈（Probabilities）
probs = logits_per_image.softmax(dim = 1)

# 遍歷所有標籤，印出每個類別對應的機率值（保留小數點後四位）
for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob.item():.4f}")

# 找出機率最高（最相似）的類別索引（Index）
best_idx = probs.argmax().item()

# 根據索引從 labels 清單中取出對應名稱，並顯示最終預測結果
print("預測:", labels[best_idx])