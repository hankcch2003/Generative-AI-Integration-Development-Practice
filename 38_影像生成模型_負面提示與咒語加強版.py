from diffusers import StableDiffusionPipeline
from deep_translator import GoogleTranslator

# 定義翻譯函式：將繁體中文翻譯為英文，以獲得最佳的模型生成效果
def zh_to_en(text):
    return GoogleTranslator(source = 'zh-TW', target = 'en').translate(text)

# 設定模型路徑並載入預訓練權重
# 在 CPU 環境下執行時，維持預設精度以避免相容性錯誤
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# 將模型移動到 CPU 進行運算
pipe = pipe.to("cpu")

# 定義中文描述，並翻譯為英文提示詞，以獲得最佳的模型生成效果
zh_prompt = "一隻在月球上的貓，寫實風格"
en_prompt = zh_to_en(zh_prompt)

# 疊加高品質關鍵字（咒語補強），提升圖片細節與光影表現
en_prompt += ", masterpiece, best quality, ultra detailed, 4k, cinematic lighting"

# 輸出最終合成的英文提示詞，確認翻譯與補強結果
print("英文 prompt:", en_prompt)

# 執行圖片生成流程
# num_inference_steps = 30：設定計算步數為 30，提高圖片生成品質
# guidance_scale = 8.5：設定引導尺度為 8.5，數值越高生成的圖片越貼近提示詞
# negative_prompt：設定負面提示詞，強制要求 AI 避開模糊、低畫質及肢體畸形
# .images[0]：從生成的結果中取出第一張圖片
image = pipe(en_prompt, num_inference_steps = 30, guidance_scale = 8.5, negative_prompt = "blurry, low quality, bad anatomy").images[0]

# 將生成的圖片儲存到當前資料夾，檔名為 38_影像生成結果_output5.png
image.save("38_影像生成結果_output5.png")