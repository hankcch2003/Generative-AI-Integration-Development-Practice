import warnings
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

# 忽略警告訊息（例如版本相容性警告），讓輸出畫面保持乾淨
warnings.filterwarnings("ignore")

# 載入處理器（負責調整圖片大小與標準化）與影像分類模型本體
# 這裡使用的是 dandelin 的 Vilt 模型，專門針對視覺問答任務進行微調
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# 指定圖片的絕對路徑（使用 r 避免反斜線被誤認為轉義字元）
image_path = r"E:\Python\Python4-人工智慧整合開發實務\20260409\images1\img2.png"

# 使用 PIL 讀取圖片，並轉換為 RGB 標準格式（避免灰階或透明圖層造成錯誤）
raw_image = Image.open(image_path).convert('RGB')

# 定義一個問題，詢問圖片中的人正在做什麼
text = "What are the people in the picture doing?"

# 將圖片處理成張量（Tensors）並設為 PyTorch 格式（pt）
encoding = processor(raw_image, text, return_tensors = "pt")

# 取得模型輸出結果
outputs = model(**encoding)

# 找出 Logits 分數最高（信心度最高）的類別索引
idx = outputs.logits.argmax(-1).item()

# 將預測結果的索引透過模型配置中的 id2label 轉換為文字標籤
print("回答:", model.config.id2label[idx])