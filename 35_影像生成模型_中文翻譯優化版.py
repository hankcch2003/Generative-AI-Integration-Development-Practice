from diffusers import StableDiffusionPipeline
from deep_translator import GoogleTranslator

# 設定模型在 Hugging Face 上的路徑（ID）
model_id = "runwayml/stable-diffusion-v1-5"

# 載入模型預訓練權重
# 在 CPU 環境下執行時，維持預設精度以避免相容性錯誤
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# 將模型移動到 CPU 進行運算
pipe = pipe.to("cpu")

# 建立翻譯器實例：設定從「繁體中文 (zh-TW)」翻譯至「英文 (en)」
translator = GoogleTranslator(source = 'zh-TW', target = 'en')

# 讓使用者在終端機輸入想要生成的中文內容描述
inputword = input('請輸入要產生的圖像描述文字：')

# 將輸入的中文翻譯成英文，因為 Stable Diffusion 模型對英文指令的反應最準確
english_prompt = translator.translate(inputword)
print('翻譯為:', english_prompt)

# 執行圖片生成流程
# num_inference_steps = 30：設定計算步數為 30，提高圖片生成品質
# .images[0]：從生成的結果中取出第一張圖片
image = pipe(english_prompt, num_inference_steps = 30).images[0]

# 將生成的圖片儲存到當前資料夾，檔名為 35_影像生成結果_output2.png
image.save("35_影像生成結果_output2.png")