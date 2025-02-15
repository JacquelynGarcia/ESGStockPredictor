# Green Signals: ESG Factors in Machine Learning Stock Price Prediction

## Project Overview
Our project explored how Environmental, Social, and Governance scores can be used to alongside machine learning to make stock price predictions. There is an increasing demand for socially responsible investing which has made ESG criteria an important consideration for financial analysis. We investigated whether combining historical financial data with ESG metrics can enhance predictive performance while promoting sustainable investment practices.

A final presentation summarizing our findings can be accessed [here](https://drive.google.com/file/d/1iO2jaRYafkq5E5xeMubC8wo6cSkyZMcQ/view?usp=sharing).

## Authors
- Auritro Dutta  
- Jacquelyn Garcia  
- Prabhmeet Gujral  
- Ethan Heath  
- Aniruddh Krovvidi  

## Dataset & Methodology
- Data was collected from Yahoo Finance (`yfinance`) and other ESG sources.
- The dataset includes historical financial indicators and ESG scores.
- Various machine learning models were implemented to analyze stock price movements:
  - **Linear Regression**
  - **Ridge Regression**
  - **Random Forest Regressor**
  - **Gradient Boosting Regressor**
- Feature engineering techniques were applied, including standard scaling and data splitting.

## Dependencies
To run this project, install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Key Libraries Used:
```python
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
```

## Running the Notebook
To execute the notebook, follow these steps:
1. Open `FinalProject_Group025_FA24.ipynb` in Jupyter Notebook.
2. Run the cells in sequence to preprocess the data, train models, and evaluate their performance.
3. Visualizations and results will be generated in the later sections.