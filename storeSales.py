import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm 
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Set plot style and size
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
df = pd.read_csv('21-Days-21-Projects-Dataset/Datasets/airline_passenger_timeseries.csv')
print(df.head())
print(df.info())
# so we have a time series data with a date column and a passenger column
# lets do some exploratory data analysis
# df.plot()
# plt.title('Airline Passenger Timeseries')
# plt.xlabel('Date')
# plt.ylabel('Passengers')
# plt.show()

# we want to change data up a bit to make it more stationary
# lets try remove months July and August
# so we need to get rid of rows with 07 and 08 in the date column
df['Month'] = pd.to_datetime(df['Month'])
df = df[~df['Month'].dt.month.isin([7, 8])]
df.plot(x='Month', y='Passengers')
plt.title('Airline Passenger Timeseries')
plt.xlabel('Date')
plt.ylabel('Passengers')

df = df.set_index('Month')
df = df.asfreq('MS')
df.loc[df.index.month.isin([7,8]), 'Passengers'] = np.nan
df['Passengers'] = df['Passengers'].interpolate('time') 
# Decompose the time series to visualize its components
decomposition = sm.tsa.seasonal_decompose(df['Passengers'], model='multiplicative', period=10)

fig = decomposition.plot()
fig.set_size_inches(12, 8)


# we can see that the data is not stationary, to use an ARIMA model we need to make it stationary
# why? As stationary data is easier to model and predict, if stationary, it means that the data is not changing over time
# therfore past data is useful for predicting future data
# to test we can use the Augmented Dickey-Fuller test, null hypothesis is that the data is not stationary
# if p-value is less than 0.05, we can reject the null hypothesis and say that the data is stationary
# we compare p value with test 0.05 or test statistic with critical values, both used for same purpose
# if p-value is less than 0.05, we can reject the null hypothesis and say that the data is stationary
# if p-value is greater than 0.05, we cannot reject the null hypothesis and say that the data is not stationary
# we can see that the data is not stationary, so we need to make it stationary
def test_stationarity(timeseries):
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationarity(df['Passengers'])
# if p-value is less than 0.05, we can reject the null hypothesis and say that the data is stationary
# if p-value is greater than 0.05, we cannot reject the null hypothesis and say that the data is not stationary
# we can see that the data is not stationary, so we need to make it stationary
# right so how do we make it stationary?
# we use log transformation and difference the data
# 1. Apply log transformation to stabilize the variance
df_log = np.log(df['Passengers'])

# 2. Apply differencing to remove the trend
df_diff = df_log.diff().dropna()

# Plot the stationary series
df_diff.plot()
plt.title('Stationary Time Series (Log-Differenced)')


# Retest for stationarity
test_stationarity(df_diff)
# plt.show()

# We now need to choose parameters for the ARIMA model
# we can use the ACF and PACF plots to help us choose the parameters
# ACF plot shows the correlation of the series with its lags
# PACF plot shows the correlation of the series with its lags, but only the direct lags
# we look for the first lag that is significant(ACF) this is our p value
# we look for the first lag that is significant(PACF) this is our q value
# its significant if it is outside the confidence interval(blanket around the line)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(df_diff, ax=ax1, lags=20)
plot_pacf(df_diff, ax=ax2, lags=20)
# plt.show() 
# from the plots we can see that the first lag that is significant is 2 for both ACF and PACF
# Split data into training and test sets
train_data = df_log[:'1958']
test_data = df_log['1959':]

# Build ARIMA model
model = ARIMA(train_data, order=(2, 1, 2), freq='MS')
arima_result = model.fit()

# Get forecast
forecast = arima_result.get_forecast(steps=len(test_data))
forecast_ci = forecast.conf_int()

# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(df_log, label='Original Log Data')
plt.plot(forecast.predicted_mean, label='Forecast')
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='k', alpha=.15)
plt.title('ARIMA Model Forecast')
plt.legend()
plt.show()