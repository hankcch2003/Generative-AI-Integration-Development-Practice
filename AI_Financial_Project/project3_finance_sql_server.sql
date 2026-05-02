-- CREATE DATABASE / 建立資料庫
CREATE DATABASE FinanceDB;
GO

-- USE DATABASE / 使用資料庫
USE FinanceDB;
GO

-- TABLE: market_data（市場資料）
CREATE TABLE market_data (
    day INT PRIMARY KEY,     -- trading day / 交易日
    price FLOAT NOT NULL,    -- market price / 市場價格
    [return] FLOAT NOT NULL  -- return rate / 報酬率
);
GO

-- TABLE: stock_info（股票資訊）
CREATE TABLE stock_info (
    stock_id INT PRIMARY KEY,  -- stock id / 股票ID
    stock_name VARCHAR(50)     -- stock name / 股票名稱
);
GO

-- TABLE: stock_price（股票價格）
CREATE TABLE stock_price (
    id INT PRIMARY KEY,  -- record id / 紀錄ID
    stock_id INT,        -- foreign key / 外鍵
    price FLOAT,         -- price / 價格

    FOREIGN KEY (stock_id) REFERENCES stock_info(stock_id)
);
GO

-- INSERT INTO / 資料插入
INSERT INTO market_data VALUES (1, 100, 0.01);
INSERT INTO market_data VALUES (2, 102, 0.02);
INSERT INTO market_data VALUES (3, 101, -0.01);
INSERT INTO market_data VALUES (4, 105, 0.03);
INSERT INTO market_data VALUES (5, 107, 0.02);

INSERT INTO stock_info VALUES (1, 'AAPL');
INSERT INTO stock_info VALUES (2, 'TSLA');
INSERT INTO stock_info VALUES (3, 'MSFT');

INSERT INTO stock_price VALUES (1, 1, 180);
INSERT INTO stock_price VALUES (2, 2, 250);
INSERT INTO stock_price VALUES (3, 3, 320);

-- BASIC QUERY / 基礎查詢
SELECT * FROM market_data;

SELECT *
FROM market_data
WHERE price > 102;

SELECT *
FROM market_data
ORDER BY price DESC;

-- JOIN / 資料關聯分析
SELECT 
    si.stock_name,
    sp.price
FROM stock_price sp
JOIN stock_info si
    ON sp.stock_id = si.stock_id;

-- Aggregation metrics for market data / 市場資料統計指標

SELECT 
    AVG(price) AS avg_price,
    AVG([return]) AS avg_return,
    SUM([return]) AS total_return,
    STDEV([return]) AS volatility
FROM market_data;

-- VIEW (Reporting Layer) / 報表層

-- 1. Daily market summary view / 日市場資料視圖
CREATE VIEW v_market_summary AS
SELECT 
    day,
    price,
    [return]
FROM market_data;

-- 2. Market KPI view / 市場關鍵指標視圖
CREATE VIEW v_market_kpi AS
SELECT 
    AVG(price) AS avg_price,
    AVG([return]) AS avg_return,
    SUM([return]) AS total_return
FROM market_data;

-- CTE for financial analysis / 金融分析暫存計算
-- Compute average market price / 計算市場平均價格

WITH avg_cte AS (
    SELECT AVG(price) AS avg_price
    FROM market_data
)
SELECT * FROM avg_cte;

-- Window Function / 時間序列分析
-- Used for time-series analytics without collapsing rows / 用於時間序列分析且不合併資料列
-- Moving average (3-day window) / 移動平均（3期）
-- Running total / 累積總和

SELECT 
    day,
    price,
    AVG(price) OVER (
        ORDER BY day
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS moving_avg
FROM market_data;

SELECT 
    day,
    price,
    SUM(price) OVER (ORDER BY day) AS running_total
FROM market_data;