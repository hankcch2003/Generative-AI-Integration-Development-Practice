from diffusers import StableDiffusionPipeline

# 設定模型在 Hugging Face 上的路徑（ID）
model_id = "runwayml/stable-diffusion-v1-5"

# 載入模型預訓練權重
# 在 CPU 環境下執行時，維持預設精度以避免相容性錯誤
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# 將模型移動到 CPU 進行運算
pipe = pipe.to("cpu")

# 定義你想生成的圖片描述 (提示詞)
# 這裡使用英文描述，模型能最精準地理解構圖需求
prompt = "A cute cat sitting on a chair, Studio Ghibli style, soft lighting, cozy atmosphere"

# 執行圖片生成流程
# num_inference_steps = 30：設定計算步數為 30，提高圖片生成品質
# .images[0]：從生成的結果中取出第一張圖片
image = pipe(prompt, num_inference_steps = 30).images[0]

# 將生成的圖片儲存到當前資料夾，檔名為 34_影像生成結果_output1.png
image.save("34_影像生成結果_output1.png")