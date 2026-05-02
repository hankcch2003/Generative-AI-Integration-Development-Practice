import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import requests

# 設定中文字型（避免中文亂碼）
matplotlib.rcParams["font.family"] = "Microsoft JhengHei"

# 避免負號顯示成方塊
matplotlib.rcParams["axes.unicode_minus"] = False

# 頁面標題
st.title("Financial AI Analytics System")

# CSV資料上傳
st.subheader("CSV 資料上傳")

uploaded_file = st.file_uploader("上傳 CSV 檔案", type = ["csv"])

df = None

# 如果有上傳檔案就讀取
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding = "utf-8")
    st.success("上傳成功")

# 如果有資料才執行分析
if df is not None:

    # 顯示資料
    st.subheader("資料預覽")
    st.dataframe(df)

    # 計算報酬率
    if "price" in df.columns:
        df["return"] = (df["price"].pct_change().replace([np.inf, -np.inf], 0).fillna(0))

    # 計算移動平均
    if "price" in df.columns:
        df["ma20"] = df["price"].rolling(20).mean().bfill()

    # 基本統計
    avg_price = df["price"].mean()
    avg_return = df["return"].mean()
    volatility = df["return"].std()

    # 顯示指標
    st.subheader("主要指標")
    st.write("平均價格：", round(avg_price, 2))
    st.write("平均報酬：", round(avg_return, 6))
    st.write("波動率：", round(volatility, 6))

    # 價格走勢圖
    st.subheader("價格走勢")

    fig, ax = plt.subplots()
    ax.plot(df["price"], label = "價格")

    if "ma20" in df.columns:
        ax.plot(df["ma20"], label = "移動平均")

    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # 報酬分布
    st.subheader("報酬分布")

    fig2, ax2 = plt.subplots()
    sns.histplot(df["return"], bins = 20, ax = ax2)
    st.pyplot(fig2)
    plt.close(fig2)

    # 機器學習預測
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["price"].values

    model = LinearRegression()
    model.fit(X, y)

    pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, pred))

    next_price = float(model.predict([[len(df)]])[0])

    # 預測結果
    st.subheader("價格預測")
    st.write("下一期預測價格：", round(next_price, 2))
    st.write("RMSE誤差：", round(rmse, 4))

    # 本地 LLM 模型名稱（Ollama）
    model_name = "llama3.2:latest"

    def call_llm(prompt):
        try:
            res = requests.post(
                "http://localhost:11434/api/generate",
                json = {
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout = 60  # 設定超時時間，避免等待過久
            )

            data = res.json()
            return data.get("response", "無回應")

        except:
            return "AI服務異常或未啟動"

    # AI提示內容（優化版）
    prompt = f"""
    你是一位專業金融數據分析師，負責分析市場數據並提供投資決策建議。

    請根據以下數據進行分析：

    平均價格: {avg_price:.2f}
    平均報酬: {avg_return:.6f}
    波動率: {volatility:.6f}

    請依照以下格式輸出：

    【市場風險評估】
    說明目前市場波動與風險程度（低/中/高），並簡短解釋原因。

    【投資建議】
    說明目前是否適合投資（進場 / 觀望 / 減碼），並提供風險控制建議。

    【策略總結】
    用一句話總結整體市場判斷與建議方向。
    """

    # AI分析區
    st.subheader("AI 投資分析")

    if st.button("產生 AI 分析"):
        result = call_llm(prompt)
        st.success(result)

else:
    st.info("請上傳 CSV 檔案")