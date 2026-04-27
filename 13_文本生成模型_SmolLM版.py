from transformers import pipeline

# 指定使用 text-generation 模型來進行文本生成，並選擇 HuggingFaceTB/SmolLM-135M 作為基礎模型
# 此模型將根據輸入的文本內容生成相關的文本
classifier = pipeline('text-generation', model = 'HuggingFaceTB/SmolLM-135M')

# 傳入起始文本，模型會根據這段內容自動續寫，並限制生成的最大字數為 100 個字
result = classifier("I love Taiwan", max_new_tokens = 100)

# 輸出模型生成的文本結果，模型會以 "I love Taiwan" 為起點自動續寫
# 生成的文本將會與原始輸入相關聯，並且可能包含對台灣的描述、情感表達或其他相關內容
print(result)