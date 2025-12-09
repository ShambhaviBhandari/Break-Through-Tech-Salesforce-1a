# 📊 Stock Price Prediction from Financial News  

## Project Description  
Our team uses publicly available financial news headlines and historical stock market data to analyze and predict short-term stock price movements. Leveraging machine learning techniques such as sentiment analysis, feature engineering, and deep learning models—including pre-trained language models (LLMs) like FinBERT—you will build a pipeline that extracts market signals from news content. This will help address the real-world challenge of interpreting financial news for timely investment decisions.  

Our main objectives are to clean and preprocess textual data, apply LLMs for sentiment or embedding extraction, align it with market data, and build a predictive model to forecast next-day stock trends. One approach could be to treat this as a classification problem, though other strategies are possible.  

---

## Data
- Financial News Headline from Kaggle: https://www.kaggle.com/datasets/notlucasp/financial-news-headlines
- yfinance: https://github.com/ranaroussi/yfinance
  
---

## Brief Overview  
This project explores the intersection of **financial news and market prediction**, aiming to forecast **short-term stock price movements** using **machine learning** and **large language models (LLMs)**.  

Our pipeline integrates **news sentiment analysis** with **historical stock data** to extract meaningful signals and deliver predictions that could support **timely and informed investment decisions**.  

---

## Objectives  
-  **Preprocess** financial news text and align it with stock market data.  
-  **Apply LLMs (e.g., FinBERT)** for sentiment scoring and embedding extraction.  
- **Engineer features** that capture market sentiment and trading signals.  
- **Train predictive models** (classification & deep learning) to forecast next-day price trends.  
-  **Evaluate model performance** using accuracy, precision, recall, and related metrics.  

---

## Approach  
1. Problem Definition
- Goal: Predict the next-day opening price of the S&P 500.
- Business Question: Can financial news sentiment combined with historical market data improve short-term stock price prediction?

2. Data Collection
- Retrieved historical S&P 500 price data using yfinance.
- Collected financial news headlines for corresponding dates.

3. Data Preparation & Integration
- Aligned historical price data and news headlines by date.
- Applied FinBERT to extract sentiment probabilities (positive, negative, neutral).
- Computed daily sentiment scores.

4. Feature Engineering
- Generated lagged price-based features:
- Previous day returns
- Volatility
- Moving averages (7-day, 30-day)
- Combined sentiment features + price features into a single dataset.

5. Modeling
- Trained several models, focusing on XGBoost for predictive performance.
- Performed model tuning and comparative evaluation.

6. Evaluation
- Evaluated using accuracy and additional regression error metrics (e.g., MAE, MSE).
- Analyzed whether sentiment improved predictions compared to using price data alone.

---

## Tech Stack  (subject to change)
- **Languages & Libraries:** Python, Pandas, NumPy, Scikit-learn, PyTorch/TensorFlow  
- **LLMs:** FinBERT, pre-trained embedding models  
- **Visualization:** Matplotlib, Seaborn  
- **Environment:** VS Code 

---

## Collaborators  👩‍💻
 **Shambhavi Bhandari**  
 **Karla Nguyen**  
 **Judy Ojewia**  
 **Chris Chen**

**Challenge Advisor:** *Atena Sadeghi*  

---

## Impact  
By combining **natural language processing** with **financial data modeling**, this project highlights how **AI-driven insights** can enhance decision-making in dynamic markets. Our approach demonstrates:  

- The potential of **news sentiment** as a predictive signal.  
- How **LLMs can be integrated** with structured financial data.  
- Real-world applications for **investors, analysts, and financial technology platforms**.  

---

✅ *This repository represents our ongoing work with Salesforce to build data-driven solutions for real-world financial challenges.*  
