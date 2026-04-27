import os
import warnings
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN

# 忽略警告訊息（例如版本相容性警告），讓輸出畫面保持乾淨
warnings.filterwarnings("ignore")

# 載入處理器（負責圖片縮放、標準化與文字 Tokenize）與 CLIP 模型
# 這裡使用的是 OpenAI 的 CLIP 模型（ViT-B/32），常用於圖文比對、影像檢索或零樣本分類
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# 定義一個函式，用於提取圖片的特徵向量（Embedding）
def get_embedding(image_path):
    # 使用 PIL 讀取圖片並轉為 RGB 標準格式（避免灰階或透明圖層造成錯誤）
    image = Image.open(image_path).convert("RGB")

    # 將圖片處理成張量（Tensors）並設為 PyTorch 格式（pt）
    inputs = processor(images = image, return_tensors = "pt")

    # 關閉梯度計算以提取影像特徵（此動作不涉及模型訓練）
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    # 判斷輸出屬性並取得影像特徵向量
    if hasattr(outputs, "pooler_output"):
        emb = outputs.pooler_output
    else:
        emb = outputs

    # 轉為 NumPy 陣列並攤平為一維向量
    emb = emb.cpu().numpy().flatten()

    # 計算向量範數（Norm）用於後續歸一化處理
    norm = np.linalg.norm(emb)

    # 進行歸一化處理，確保特徵向量長度一致以利相似度計算
    if norm > 1e-9:
        emb = emb / norm        
    return emb

# 指定影像資料夾目錄路徑（使用 r 避免反斜線轉義錯誤）
image_dir = r'E:\Python\Python4-人工智慧整合開發實務\20260409\images1'

# 批量合成完整圖片路徑清單（使用 os.path.join 串接路徑與檔名）
paths = [os.path.join(image_dir, p) for p in os.listdir(image_dir)]

# 提取所有圖片的特徵向量（Embeddings）並存入清單中以利後續運算
embeddings = [get_embedding(p) for p in paths]

# 初始化一個空的清單（List），用於存放後續偵測到的重複或高度相似圖片資訊
duplicates = []

# 第一階段：檢測重複或高度相似圖片
# 使用雙重迴圈遍歷所有圖片特徵，進行兩兩比對
for i in range(len(paths)):
    for j in range(i + 1, len(paths)):

        # 計算兩兩影像間的餘弦相似度（Cosine Similarity）
        sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]   

        # 若相似度高於 0.95 則判定為重複或高度相似
        if sim > 0.95:  
            # 將符合條件的圖片路徑對與相似度分數存入重複清單（Duplicates）
            duplicates.append((paths[i], paths[j], sim))

# 遍歷重複清單並輸出結果
for d in duplicates:
    print("重複:", d)

# 第二階段：使用 K-Means 演算法進行固定分群
k = 3  # 設定預計分成的群數（Cluster）

# 初始化 K-Means 模型並設定群數與隨機種子（確保每次執行結果相同）
kmeans = KMeans(n_clusters = k, random_state = 0)

# 將提取的特徵向量進行分群計算，並取得每張圖片對應的群組標籤（Labels）
labels = kmeans.fit_predict(embeddings)

print('分成三群')

# 遍歷所有圖片路徑與其對應的分群標籤（使用 zip 進行配對）
for path, label in zip(paths, labels):
    # 輸出每張圖片所屬的分群編號（Label）與其完整路徑
    print(label, path)

# 第三階段：使用 DBSCAN 演算法進行自動分群
# 初始化 DBSCAN 模型並設定半徑門檻（eps）、最小樣本數（min_samples）與餘弦相似度度量基準
clustering = DBSCAN(eps = 0.3, min_samples = 2, metric = "cosine")

# 根據特徵向量的密度自動進行分群（不需要預先指定群數）
labels = clustering.fit_predict(embeddings)

print('不設定群數（標記為 -1 代表雜訊或無相關群體）')

# 遍歷所有圖片路徑與自動分群後的標籤清單
for path, label in zip(paths, labels):
    # 顯示分群編號與對應圖片的路徑
    print(label, path)