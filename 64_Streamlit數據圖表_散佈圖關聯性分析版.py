import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# 設定中文字型與符號
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

# 標題
st.title("散佈圖示範：內建 vs 可客製化")

# 資料
df = pd.DataFrame({"身高": [160, 175, 190], "體重": [90, 80, 75]})

# 方法一：Streamlit 內建散佈圖
st.subheader("圖表一：st.scatter_chart（快速）")
st.write("優點：簡單快速；缺點：無法控制細節（如座標軸文字、標題）")
st.scatter_chart(df.set_index("身高"))

st.divider()

# 方法二：Matplotlib（可客製化）
st.subheader("圖表二：matplotlib（可客製化）")
st.write("優點：可調整圖表細節，如標題、座標軸、字型、標記樣式")

# 建立圖表
fig, ax = plt.subplots(figsize = (5, 3))

# 畫散佈圖，設定點顏色與大小
ax.scatter(df["身高"], df["體重"], color = "blue", s = 100)  # s: 點大小, color: 顏色

# 設定標題與軸標籤
ax.set_title("身高 vs 體重")
ax.set_xlabel("身高 (cm)")
ax.set_ylabel("體重 (kg)")

# 控制 x 軸文字水平
ax.tick_params(axis = 'x', rotation = 0)

# 控制 y 軸文字水平，並偏移距離避免貼近圖表
ax.set_ylabel("體重 (kg)", rotation = 0, labelpad = 25)

# 自動調整圖表布局
fig.tight_layout()

# 在 Streamlit 中顯示
st.pyplot(fig, use_container_width = True)