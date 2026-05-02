# AI Financial Analytics Projects

本專案整合三個金融數據分析子系統，  
結合區塊鏈與股票市場分析，並導入機器學習與 AI 模型，  
用於投資決策輔助與風險評估。

---

## 🧠 系統概述

本專案以「數據驅動投資分析」為核心，包含以下四大模組：

- 📊 市場數據模擬層（Crypto / Stock Data Simulation）
- 📈 數據分析層（Technical Indicators & Risk Analysis）
- 🤖 機器學習預測層（Linear Regression Price Prediction）
- 🧠 AI 決策輔助層（LLM Investment Advice）

---

## 🛠 技術架構

- Python
- Streamlit
- Pandas / NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Ollama LLM

---

## 🎯 專案目標

透過 AI 與金融數據分析技術，  
建立一個可視化投資分析系統，  
協助使用者進行：

- 投資風險判斷
- 市場趨勢分析
- 價格預測
- AI 輔助決策

---

# 📊 Project 1: blockchain_ai.py

## 專案說明

本系統模擬區塊鏈（加密貨幣）市場價格，  
並結合資料分析、機器學習與 AI 模型，  
建立完整的金融數據分析與投資決策輔助系統。

## 功能模組

- 📈 價格趨勢分析（Price Trend）
- 📊 報酬率分布分析（Return Distribution）
- 🤖 機器學習預測（Linear Regression）
- 📉 模型評估（RMSE）
- 🧠 AI 投資建議（Ollama LLM）

---

# 📈 Project 2: stock_risk.py

## 專案說明

本系統模擬股票市場資料，  
分析報酬與風險之間的關係，  
並建立投資效率與風險分類模型。

## 功能模組

- 📊 股票報酬與波動分析
- ⚠️ 風險分類（High / Medium / Low）
- 📉 風險 vs 報酬視覺化
- 📈 Sharpe Ratio（投資效率指標）

---

# 📊 Project 3: financial_ai_system.py

## 專案說明

本系統結合真實 CSV 金融資料分析、技術指標計算、機器學習預測與 AI 大型語言模型（LLM），  
建立一個互動式金融數據分析平台（Streamlit Dashboard），用於投資分析與決策輔助。

---

## 功能模組

### 📂 數據輸入
- CSV 檔案上傳
- 即時資料讀取與顯示

---

### 📈 技術分析
- 報酬率計算（Return）
- 移動平均（MA20）
- 波動率分析（Volatility）

---

### 📉 視覺化分析
- 價格走勢圖
- 報酬率分布圖

---

### 🤖 機器學習預測
- 線性回歸模型
- 價格預測
- RMSE 誤差評估

---

### 🧠 AI 投資分析
- Ollama LLM 串接
- 自動生成投資建議
- 風險評估與策略分析

---

## 🗄️ Database Design（資料庫設計）

本 SQL 模組隸屬於 Project 3，  
用於展示金融資料結構規劃，並支援分析系統運作。

---

### 📊 Tables

#### 📌 stock_info（股票基本資料表）
- stock_id：股票代碼（Primary Key）
- stock_name：股票名稱

---

#### 📌 stock_price（股票價格資料表）
- id：資料編號（Primary Key）
- stock_id：股票代碼（Foreign Key）
- price：股票價格

---

#### 📌 market_data（市場數據示意表）
- date：交易日期
- price：市場價格
- return：報酬率

---

### 🔗 Relationships

- stock_info (1) ─── (N) stock_price  
- stock_price.stock_id → stock_info.stock_id

---

## 🧩 ER Model（實體關聯模型）

### 📊 Entities

- market_data  
  市場時間序列資料（Time Series Market Dataset）

- stock_info  
  股票主資料表（Stock Master Table）

- stock_price  
  股票價格紀錄（Stock Price Records）

---

### 🔗 Relationships

- stock_info (1) ─── (N) stock_price  

---

### 🔑 Foreign Key Constraint

- stock_price.stock_id → stock_info.stock_id

---

## 🧠 Design Concept

- 🗄️ SQL 層：資料結構設計與關聯規劃
- 🐍 Python 層：數據分析與 AI 預測
- 🖥️ Streamlit 層：視覺化介面

---

## 📌 資料來源說明

SQL 與 Python 使用不同資料來源：

- 🗄️ SQL 用於資料結構設計與關聯展示  
- 🐍 Python 使用 CSV / 模擬資料進行即時分析與模型訓練  

👉 兩者為分層架構設計，並非直接共享同一份資料

---

## 📌 系統特色

- CSV 即時上傳分析
- 技術指標 + ML + AI 整合
- 互動式金融儀表板
- 模組化系統設計

---

## 📌 備註

本專案所有數據皆為模擬資料，  
用於展示 AI + 金融分析流程與系統架構設計。