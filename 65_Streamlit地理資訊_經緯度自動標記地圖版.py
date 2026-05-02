import pandas as pd
import streamlit as st

# 標題
st.title("地圖示範：st.map")

# 說明
st.write("st 提供簡單的地圖功能，可以用經緯度資料直接畫地圖標記")

# 資料：經緯度
df = pd.DataFrame({
    "lat": [25.053680, 24.348599, 23.113986, 31.616140],
    "lon": [121.519471, 120.760597, 121.208105, 130.675008]
})

# 顯示地圖
st.map(df)