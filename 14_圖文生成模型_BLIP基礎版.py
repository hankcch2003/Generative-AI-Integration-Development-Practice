import warnings
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 忽略警告訊息（例如版本相容性警告），讓輸出畫面保持乾淨
warnings.filterwarnings("ignore")

# 設定模型 ID，這裡使用的是 Salesforce 的 BLIP 基礎圖文生成模型（Base 版本）
model_id = "Salesforce/blip-image-captioning-base"

# 載入處理器（負責把圖片或文字轉換成模型看得懂的數字）與模型本體
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id)

# 指定圖片的絕對路徑（使用 r 避免反斜線被誤認為轉義字元）
image_path = r"E:\Python\Python4-人工智慧整合開發實務\20260409\images1\img3.png"

# 使用 PIL 讀取圖片，並轉換為 RGB 標準格式（避免灰階或透明圖層造成錯誤）
raw_image = Image.open(image_path).convert('RGB')

# 將圖片處理成張量（Tensors）並設為 PyTorch 格式（pt）
inputs = processor(raw_image, return_tensors = "pt")

# 讓模型生成描述（由模型針對圖檔特徵進行解析）
out = model.generate(**inputs)

# 將模型輸出的數字解碼回人類看得懂的文字，並移除特殊標記字元
caption = processor.decode(out[0], skip_special_tokens = True)

# 顯示生成的結果
print("Image Caption:", caption)