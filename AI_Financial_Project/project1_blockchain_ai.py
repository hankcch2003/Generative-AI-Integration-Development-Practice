import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import requests

# 頁面標題
st.title("Blockchain AI Financial System")

# 本地 LLM 模型名稱（Ollama）
model = "llama3.2:latest"

# 模擬時間長度
days = 120

# 固定隨機種子（確保結果一致）
np.random.seed(42)

# 模擬市場趨勢（線性成長 + 隨機波動）
trend = np.linspace(0, 10, days)
noise = np.random.randn(days) * 2
price = 100 + trend + np.cumsum(noise)

# 建立價格資料表
df = pd.DataFrame({
    "Day": np.arange(days),
    "Price": price
})

# 補齊缺失值（避免計算錯誤）
df["Price"] = df["Price"].ffill()

# 計算技術指標（移動平均）
df["MA20"] = df["Price"].rolling(20).mean()

# 計算報酬率（價格變化幅度）
df["Return"] = df["Price"].pct_change().fillna(0)

# 價格趨勢視覺化
st.subheader("Price Trend")

fig, ax = plt.subplots()
ax.plot(df["Day"], df["Price"], label = "Price")  # 原始價格
ax.plot(df["Day"], df["MA20"], label = "MA20")    # 移動平均
ax.legend()

st.pyplot(fig)
plt.close(fig)  # 釋放記憶體

# 報酬率分布分析（風險觀察）
st.subheader("Return Distribution")

fig2, ax2 = plt.subplots()
sns.histplot(df["Return"], bins = 30, ax = ax2)

ax2.set_xlabel("Return")
ax2.set_ylabel("Frequency")

st.pyplot(fig2)
plt.close(fig2)

# 機器學習模型（價格預測）
st.subheader("Price Prediction")

# 特徵（時間）
X = df["Day"].values.reshape(-1, 1)

# 標籤（價格）
y = df["Price"].values

# 模型訓練（Linear Regression）
@st.cache_resource  # 快取模型避免重複訓練
def train_model(X, y):
    model_lr = LinearRegression()
    model_lr.fit(X, y)
    return model_lr

model_lr = train_model(X, y)

# 預測訓練集
pred = model_lr.predict(X)

# 計算誤差（RMSE）
rmse = np.sqrt(mean_squared_error(y, pred))

# 預測下一期價格
next_price = float(model_lr.predict([[days]])[0])

st.write("Next Price：", round(next_price, 2))
st.write("RMSE：", round(rmse, 4))

# LLM API 呼叫（Ollama）
def call_llm(prompt):
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json = {
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout = 30  # 防止卡住
        )
        return res.json().get("response", None)
    except Exception:
        return None  # 連線失敗或 API 錯誤

# AI 投資建議模組
st.subheader("AI Investment Advice")

# 計算市場統計值
avg_return = df["Return"].mean()
volatility = df["Return"].std()

# 給 LLM 的金融提示詞
prompt = f"""
你是一位量化金融分析師。

市場資訊：
平均報酬率：{avg_return:.6f}
波動率：{volatility:.6f}

請提供：
1. 風險判斷
2. 投資建議
3. 50字內總結
"""

# 使用者觸發 AI 分析
if st.button("Generate AI Advice"):
    result = call_llm(prompt)

    if result is None:
        st.warning("AI無回應，請確認模型是否啟動")
    else:
        st.success(result)