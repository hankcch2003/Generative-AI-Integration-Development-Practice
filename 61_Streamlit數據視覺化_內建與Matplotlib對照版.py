import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# 資料
df = pd.DataFrame({"x": range(10), "y": [i ** 2 for i in range(10)]})

# 圖表一：內建圖表（快速簡單，但無法客製化）
st.subheader("圖表一：st.line_chart（快速）")
st.line_chart(df.set_index("x"))

st.divider()

# 圖表二：matplotlib（可客製化：標題、座標軸等，但寫法較複雜）
st.subheader("圖表二：matplotlib（可客製化）")
fig, ax = plt.subplots()
ax.plot(df["x"], df["y"])
ax.set_title("y = x²")
ax.set_xlabel("x")
ax.set_ylabel("y")
st.pyplot(fig)