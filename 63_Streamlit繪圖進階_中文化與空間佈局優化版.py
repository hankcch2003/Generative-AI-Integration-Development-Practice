import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# 設定中文字型與符號，以正確顯示中文與負號
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

# 標題
st.title("圖表對照：內建 vs 可客製化（含座標軸控制）")

# 資料
df = pd.DataFrame({"科目": ["數學", "英文", "國文"], "分數": [90, 80, 75]})

# 方法一：Streamlit 內建圖表
st.subheader("圖表一：st.bar_chart（快速）")
st.write("優點：簡單快速；缺點：無法調整座標軸細節（如文字方向）")
st.bar_chart(df.set_index("科目"))

st.divider()

# 方法二：Matplotlib（可客製化）
st.subheader("圖表二：matplotlib（可客製化）")
st.write("優點：可調整圖表細節，例如座標軸文字方向、標題、字型等")

# 建立圖表與軸
fig, ax = plt.subplots(figsize = (5, 3))

# 畫長條圖
ax.bar(df["科目"], df["分數"])

# 設定標題與軸標籤
ax.set_title("成績圖")
ax.set_xlabel("科目")
ax.set_ylabel("分數")

# 控制座標軸文字方向
ax.tick_params(axis = 'x', rotation = 0)            # x 軸文字水平
ax.set_ylabel("分數", rotation = 0, labelpad = 20)  # y 軸文字水平，與圖外偏移

# 自動調整圖表布局，避免文字被截掉
fig.tight_layout()

# 在 Streamlit 中顯示圖表，寬度自動填滿容器
st.pyplot(fig, use_container_width = True)