This project builds a complete time series forecasting pipeline using the classic airline passengers dataset.

## Steps
1. **Exploration & Decomposition**  
   - Visualized the monthly passenger data.  
   - Decomposed into trend, seasonality, and residuals.  

2. **Stationarity Check**  
   - Applied the Augmented Dickey-Fuller (ADF) test.  
   - Used log transform + differencing to achieve stationarity.  

3. **Modeling**  
   - **ARIMA**: captured trend but failed to capture strong seasonality.  
   - **SARIMA**: modeled both trend and seasonality effectively.  

4. **Evaluation**  
   - Train/test split (train up to 1958, test on 1959).  
   - Forecast accuracy measured with RMSE.  

## Key Takeaways
- Stationarity is crucial for ARIMA-family models.  
- Seasonal ARIMA (SARIMA) significantly outperforms plain ARIMA when seasonality is present.  
- Dropping months changes the seasonal period (12 → 10), so either adapt the model or impute missing data.  
