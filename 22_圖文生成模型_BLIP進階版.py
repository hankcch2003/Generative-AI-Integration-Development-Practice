import warnings
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# 忽略警告訊息（例如版本相容性警告），讓輸出畫面保持乾淨
warnings.filterwarnings("ignore")

# 檢查是否有 NVIDIA GPU（CUDA）可以加速，否則使用 CPU 運算
device = "cuda" if torch.cuda.is_available() else "cpu"

# 設定模型 ID，這裡使用的是 Salesforce 的 BLIP 大型圖文生成模型
model_id = "Salesforce/blip-image-captioning-large"

# 載入處理器（負責把圖片或文字轉換成模型看得懂的數字）與模型本體
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)

# 指定圖片的絕對路徑（使用 r 避免反斜線被誤認為轉義字元）
image_path = r"E:\Python\Python4-人工智慧整合開發實務\20260409\test.png"

# 使用 PIL 讀取圖片，並轉換為 RGB 標準格式（避免灰階或透明圖層造成錯誤）
raw_image = Image.open(image_path).convert('RGB')

# 模式 1：無提示生成（讓 AI 自己決定要說什麼）
# 將圖片處理成張量（Tensors）並送到指定設備（GPU 或 CPU）
inputs = processor(raw_image, return_tensors = "pt").to(device)

# 讓模型生成描述，並限制生成的最大字數為 50 個字
out = model.generate(**inputs, max_new_tokens = 50)

# 將模型輸出的數字解碼回人類看得懂的文字，並顯示生成的預測結果（移除特殊標記字元）
print("生成描述:", processor.decode(out[0], skip_special_tokens = True))

# 模式 2：有提示生成（引導 AI 從特定字句開始描述）
# 設定開頭文字，引導模型從特定詞彙開始進行後續描述
text = "a photo of"  

# 將圖片與提示文字一起處理成張量（Tensors）並送到指定設備
inputs = processor(raw_image, text, return_tensors = "pt").to(device)

# 讓模型生成描述，並限制生成的最大字數為 50 個字
out = model.generate(**inputs, max_new_tokens = 50)

# 將模型輸出的數字解碼回人類看得懂的文字，並顯示引導後的生成結果
print("有提示生成:", processor.decode(out[0], skip_special_tokens = True))