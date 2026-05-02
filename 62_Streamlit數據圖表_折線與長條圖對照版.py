import pandas as pd
import streamlit as st

# 標題
st.title("圖表對照：折線圖 vs 長條圖")

# 資料
df = pd.DataFrame({"科目": ["數學", "英文", "國文"], "分數": [90, 80, 75]})

# 折線圖
st.subheader("圖表一：折線圖（line_chart）")
st.write("適合看趨勢變化，但這裡是類別資料，因此不是最適合的圖表呈現方式")
st.line_chart(df.set_index("科目"))

st.divider()

# 長條圖
st.subheader("圖表二：長條圖（bar_chart）")
st.write("適合比較不同類別的數值大小")
st.bar_chart(df.set_index("科目"))