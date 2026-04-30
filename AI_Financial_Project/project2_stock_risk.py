import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 頁面標題
st.title("Stock Risk Dashboard")

# 固定隨機種子（確保結果可重現）
np.random.seed(7)

# 模擬股票資料（報酬率 + 波動率）
df = pd.DataFrame({
    "Stock": ["AAPL", "TSLA", "MSFT", "AMZN"],      # 股票名稱
    "Return": np.random.uniform(-0.05, 0.15, 4),    # 報酬率
    "Volatility": np.random.uniform(0.01, 0.08, 4)  # 風險（波動率）
})

# 缺失值補 0（避免資料異常）
df = df.fillna(0)

# 顯示原始資料
st.subheader("Stock Data")
st.dataframe(df)

# 風險分類函式（依波動率判斷風險等級）
def risk(v):
    if v > 0.06:
        return "High Risk"
    elif v > 0.03:
        return "Medium Risk"
    else:
        return "Low Risk"

# 建立風險分類欄位
df["Risk"] = df["Volatility"].apply(risk)

# 計算夏普比率（投資效率指標）
df["Sharpe"] = df["Return"] / (df["Volatility"] + 1e-6)

# 依夏普比率排序（投資效率由高到低）
df = df.sort_values("Sharpe", ascending = False)

# 風險與報酬視覺化
st.subheader("Risk vs Return")

fig, ax = plt.subplots()
sns.scatterplot(
    data = df,
    x = "Volatility",
    y = "Return",
    hue = "Stock",
    ax = ax
)

# 圖表標題
ax.set_title("風險與報酬關係圖")

# 顯示圖表
st.pyplot(fig)

# 分析結果輸出
st.subheader("Risk Result")

# 格式化顯示（百分比 + 小數）
st.dataframe(df.style.format({
    "Return": "{:.2%}",
    "Volatility": "{:.2%}",
    "Sharpe": "{:.2f}"
}))