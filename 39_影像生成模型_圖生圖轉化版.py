from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# 設定模型在 Hugging Face 上的路徑（ID）
model_id = "runwayml/stable-diffusion-v1-5"

# 載入模型預訓練權重
# 在 CPU 環境下執行時，維持預設精度以避免相容性錯誤
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id)

# 將模型移動到 CPU 進行運算
pipe = pipe.to("cpu")

# 指定圖片的絕對路徑（使用 r 避免反斜線被誤認為轉義字元）
init_image = r"E:\Python\Python4-人工智慧整合開發實務\20260414\2015game.jpg"

# 使用 PIL 讀取圖片，並轉換為 RGB 標準格式（避免灰階或透明圖層造成錯誤）
raw_image = Image.open(init_image).convert('RGB')

# 調整圖片大小以符合模型的輸入要求，確保圖片不會過大或過小導致處理失敗
init_image = raw_image.resize((768, 512))

# 定義正向與負向提示詞，用於引導模型生成符合預期的圖像
prompt = "A fantasy landscape, oil painting style, floating islands, high detail"
negative_prompt = "blurry, distorted, low quality"

# 執行圖片生成流程
# num_inference_steps = 30：設定計算步數為 30，提高圖片生成品質
# guidance_scale = 8.5：設定引導尺度為 8.5，數值越高生成的圖片越貼近提示詞
# negative_prompt：設定負面提示詞，強制要求 AI 避開模糊、低畫質及肢體畸形
# image = init_image：指定參考的原始圖片
# strength = 0.75：設定對原圖的改動強度為 0.75，數值範圍為 0~1，數值越高改動幅度越大
# .images：從生成的結果中取出圖片列表
images = pipe(prompt = prompt, num_inference_steps = 30, guidance_scale = 8.5, negative_prompt = negative_prompt,
              image = init_image, strength = 0.75).images

# 將生成的圖片儲存到當前資料夾，檔名為 39_影像生成結果_output6.png
images[0].save("39_影像生成結果_output6.png")