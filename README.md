# NVIDIA Stock Price Analysis

A data science project analysing NVIDIA Corporation (NVDA) stock prices from 2015 to 2024, combining historical price data with macroeconomic event data to explore patterns and build predictive regression models.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Team Members](#team-members)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Requirements](#requirements)
- [How to Run](#how-to-run)

---

## Project Overview

This project was developed as part of **CP462** (Data Science course). The goal is to:

1. Wrangle and clean NVIDIA stock price data alongside Forex Factory macroeconomic event data.
2. Perform exploratory data analysis (EDA) to uncover trends, patterns, and relationships.
3. Engineer features and train linear regression models to predict the adjusted closing price.

---

## Team Members

| Name | Student ID |
|------|-----------|
| นายรพีภัทร อุ่นคำ | 66102010152 |
| นายธนพธ์ อุ่นทรัพย์ | 66102010239 |
| นายภคพล ต้นสาลี | 66102010243 |

---

## Datasets

| Dataset | Description | Source |
|---------|-------------|--------|
| `nvidia_stock_2015_to_2024.csv` | Daily OHLCV data for NVDA (2015–2024) | [Kaggle – NVIDIA Corporation (NVDA) Stock](https://www.kaggle.com/code/fareedalianwar/nvidia-corporation-nvda-stock-2015-2024/notebook) |
| `scrape.csv` | Forex Factory economic event data (CPI, Federal Funds Rate, etc.) | [Kaggle – Forex Factory Entire Dataset](https://www.kaggle.com/datasets/randelltsen/forex-factory-entire-dataset-till-2024-08-16) |

---

## Project Structure

```
NVIDIA-Stock-Price-Analysis/
├── PROJECT_NVDA_STOCK.ipynb   # Main Jupyter Notebook
└── README.md
```

---

## Methodology

### 1. Data Wrangling

| Task | Description |
|------|-------------|
| **Gathering** | Load stock CSV and economic news CSV into DataFrames |
| **Cleaning** | Drop unused columns, handle missing values, detect negative values, and identify outliers using IQR |
| **Transforming** | Parse and normalise date columns, sort by date |
| **Enriching** | Compute 50-day & 200-day moving averages (MA50, MA200), daily return, lag features, and date components; merge news data onto stock data |
| **Validation** | Verify OHLC consistency: `low ≤ open/close ≤ high` |
| **Publishing** | Export cleaned and merged DataFrames to CSV |

### 2. Exploratory Data Analysis (EDA)

- **Univariate Analysis** – Distribution of daily returns and trading volume (histograms with KDE)
- **Bivariate Analysis** – Volume vs. daily return scatter plot; CPI surprise vs. next-day return
- **Multivariate Analysis** – Adjusted close price with MA50/MA200, pair plot, correlation heatmap
- **Pattern & Trend Detection** – Long-term price trend, volume over time, daily return volatility

### 3. Feature Engineering

Lag features from the previous trading day (`t-1`) are created for:

- `open`, `high`, `low`, `close`, `volume`, `adjclose`

Additional date features: `Day_label` (day of week), `month`, `Day`.

### 4. Machine Learning Models

All models are tuned with **5-fold cross-validated GridSearchCV**:

| Model | Notes |
|-------|-------|
| **Linear Regression** | Baseline model |
| **Ridge Regression** | L2 regularisation; polynomial features (degree 1–3); `alpha` tuned via grid search |
| **Lasso Regression** | L1 regularisation; polynomial features; `alpha` tuned via grid search |
| **Elastic Net** | L1 + L2 regularisation; polynomial features; `alpha` and `l1_ratio` tuned via grid search |

**Target variable:** `adjclose` (adjusted closing price)

---

## Results

The best-performing model was **Ridge Regression** (with polynomial features):

| Metric | Train | Test |
|--------|-------|------|
| R² | 0.998 | 0.998 |
| MSE | 0.575 | 0.503 |

The model explains ~99.8% of the variance in the adjusted closing price, with low prediction error on both the training and test sets.

---

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## How to Run

1. Clone the repository and place the dataset CSV files (`nvidia_stock_2015_to_2024.csv` and `scrape.csv`) in the same directory as the notebook.
2. Open `PROJECT_NVDA_STOCK.ipynb` in Jupyter Notebook or JupyterLab.
3. Run all cells from top to bottom (`Kernel → Restart & Run All`).
