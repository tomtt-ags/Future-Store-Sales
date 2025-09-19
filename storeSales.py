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
# Decompose the time series to visualize its components
decomposition = sm.tsa.seasonal_decompose(df['Passengers'], model='multiplicative', period=10)

fig = decomposition.plot()
fig.set_size_inches(14, 10)


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
plt.show()