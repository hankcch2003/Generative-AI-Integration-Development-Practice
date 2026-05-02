import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 設定中文字型，避免中文或符號顯示不正常
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

# 建立三個分頁
tab1, tab2, tab3 = st.tabs(["長條圖", "散佈圖", "地圖呈現"])

# 分頁 1：長條圖
with tab1:
    st.write("長條圖示範")
    
    # 建立示範資料
    df = pd.DataFrame({
        "科目": ["數學", "英文", "國文"],
        "分數": [90, 80, 75]
    })

    # 內建長條圖（快速簡單）
    st.subheader("內建長條圖")
    st.write("優點：快速、簡單；缺點：無法客製化細節")
    st.bar_chart(df.set_index("科目"))

    # Matplotlib 可客製化長條圖
    st.subheader("Matplotlib 長條圖")
    st.write("優點：可調整圖表細節，例如顏色、標題、座標軸文字方向")
    fig, ax = plt.subplots(figsize = (5, 3))

    ax.bar(df["科目"], df["分數"], color = "skyblue")   # 畫長條圖，設定顏色
    ax.set_title("成績圖")                              # 設定圖表標題
    ax.set_xlabel("科目")                               # x 軸標籤
    ax.set_ylabel("分數", rotation = 0, labelpad = 15)  # y 軸文字水平，並稍微偏移
    ax.tick_params(axis = 'x', rotation = 0)            # x 軸文字水平
    ax.tick_params(axis = 'y', rotation = 0)            # y 軸文字水平
    fig.tight_layout()                                  # 自動調整圖表布局，避免文字被切掉
    st.pyplot(fig, use_container_width = True)          # 顯示 Matplotlib 圖表

# 分頁 2：散佈圖
with tab2:
    st.write("散佈圖示範")
    
    # 建立示範資料
    df = pd.DataFrame({
        "身高": [160, 175, 190],
        "體重": [90, 80, 75]
    })

    # 內建散佈圖（快速簡單）
    st.subheader("內建散佈圖")
    st.write("優點：快速、簡單；缺點：無法控制點大小、顏色與座標軸文字")
    st.scatter_chart(df.set_index("身高"))

    # Matplotlib 可客製化散佈圖
    st.subheader("Matplotlib 散佈圖")
    st.write("優點：可調整點大小、顏色、標題、座標軸文字方向")
    fig, ax = plt.subplots(figsize = (5, 3))

    ax.scatter(df["身高"], df["體重"], color = "red", s = 100)  # 畫散佈圖，設定顏色與大小
    ax.set_title("身高 vs 體重")                                # 設定圖表標題
    ax.set_xlabel("身高 (cm)")                                  # x 軸標籤
    ax.set_ylabel("體重 (kg)", rotation = 0, labelpad = 25)     # y 軸文字水平，偏移避免靠近圖表
    ax.tick_params(axis = 'x', rotation = 0)                    # x 軸文字水平
    ax.tick_params(axis = 'y', rotation = 0)                    # y 軸文字水平
    fig.tight_layout()                                          # 自動調整圖表布局
    st.pyplot(fig, use_container_width = True)                  # 顯示 Matplotlib 圖表

# 分頁 3：地圖呈現
with tab3:
    st.write("地圖呈現示範")
    
    # 建立經緯度資料
    df = pd.DataFrame({
        "lat": [25.053680, 24.348599, 23.113986, 31.616140],
        "lon": [121.519471, 120.760597, 121.208105, 130.675008]
    })

    # Streamlit 內建地圖
    st.subheader("內建地圖")
    st.write("優點：簡單快速；缺點：無法客製化圖表細節")
    st.map(df)