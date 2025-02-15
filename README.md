# Green Signals: ESG Factors in Machine Learning Stock Price Prediction

## Project Overview
This project explores the integration of Environmental, Social, and Governance (ESG) factors into machine learning-based stock price prediction. The increasing demand for socially responsible investing has made ESG criteria an important consideration for financial analysis. We investigate whether combining historical financial data with ESG metrics can enhance predictive performance while promoting sustainable investment practices.

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

## Results & Conclusions
- The models demonstrated varying degrees of accuracy in predicting stock prices based on ESG factors.
- Feature importance analysis indicated which ESG metrics were most relevant.
- Further improvements could involve alternative ML algorithms or additional financial indicators.
