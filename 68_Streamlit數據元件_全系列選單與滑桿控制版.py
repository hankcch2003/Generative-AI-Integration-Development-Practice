import streamlit as st

# 單選下拉
city = st.selectbox("選擇城市", ["台北", "台中", "高雄", "台南"])
st.write(f"你選了：{city}")

# 多選下拉
langs = st.multiselect("選擇程式語言", ["Python", "R", "Java", "C++"])
st.write(f"你選了：{langs}")

# 數值滑桿
age = st.slider("選擇年齡", 0, 100, 25, 1)
st.write(f"年齡：{age}")

# 範圍滑桿
price_range = st.slider("價格區間", 0, 1000, (200, 800))
st.write(f"價格：{price_range[0]} ~ {price_range[1]} 元")

# 文字滑桿
grade = st.select_slider("選擇等級", options = ["差", "普通", "好", "優秀", "卓越"])
st.write(f"等級：{grade}")