import streamlit as st

# 建立三個分頁
tab1, tab2, tab3 = st.tabs(["教學介紹", "實作練習", "測驗題目"])

# 分頁 1：教學介紹
with tab1:
    st.write("這裡放教學內容")

# 分頁 2：實作練習
with tab2:
    st.write("這裡放實作練習")

# 分頁 3：測驗題目
with tab3:
    st.write("這裡放測驗題目")