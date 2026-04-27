import os
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
base_path = r"E:\Python\Python4-人工智慧整合開發實務\20260409\images1"

# 定義要進行比對的圖片檔案名稱清單
image_paths = ["img1.png", "img2.png", "img3.png", "img4.png"]

# 使用 PIL 批量讀取圖片，並轉換為 RGB 標準格式（避免灰階或透明圖層造成錯誤）
images = [Image.open(os.path.join(base_path, p)).convert("RGB") for p in image_paths]

# 定義一段文字敘述（Query），模型會從多張圖片中找出最符合這段描述的影像
query = "a person riding a bike"

# 將文字與多張圖片同時處理成張量（Tensors）並設為 PyTorch 格式（pt）
inputs = processor(text = [query], images = images, return_tensors = "pt", padding = True)

# 取得模型輸出的原始相似度分數（Logits）
outputs = model(**inputs)

# 從輸出中提取文字對圖片的原始分數（Logits）
# 這裡使用 logits_per_text 是因為我們要從一堆圖片中選出最符合該文字的結果
logits_per_text = outputs.logits_per_text

# 使用 Softmax 將原始分數轉換為總和為 1 的機率分佈（Probabilities）
probs = logits_per_text.softmax(dim = 1)

# 遍歷所有圖片路徑，並輸出每張圖片與該文字敘述的符合機率（保留小數點後四位）
for path, prob in zip(image_paths, probs[0]):
    print(f"{path}: {prob.item():.4f}")

# 找出機率最高（最符合描述）的圖片索引（Index）
best_idx = probs.argmax().item()

# 根據索引從 image_paths 清單中取出對應檔名，並顯示最終結果
print("最符合:", image_paths[best_idx])