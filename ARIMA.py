import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load FEMA data
fema_data = pd.read_csv('us_disaster_declarations.csv')

# Convert dates into datetime objects
fema_data['declaration_date'] = pd.to_datetime(fema_data['declaration_date'])

# Count the number of natural disasters each year
disasters_per_year = fema_data.groupby(fema_data['declaration_date'].dt.year).size()

# Stationarity test
result = adfuller(disasters_per_year.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

if result[1] > 0.05:
    print("Series is not stationary. Differencing may be needed.")
    disasters_per_year = disasters_per_year.diff().dropna()
    # Convert integer indices to date format (last day of each year)
disasters_per_year.index = pd.to_datetime(disasters_per_year.index, format='%Y') + pd.offsets.YearEnd()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Plot the Autocorrelation Function (ACF)
plt.figure(figsize=(10, 6))
plot_acf(disasters_per_year.dropna(), ax=plt.gca(), lags=20)  # Adjust lags as necessary
plt.title('Autocorrelation Function')

# Plot the Partial Autocorrelation Function (PACF)
plt.figure(figsize=(10, 6))
plot_pacf(disasters_per_year.dropna(), ax=plt.gca(), lags=20)  # Adjust lags as necessary
plt.title('Partial Autocorrelation Function')

plt.show()

# Attempt to fit an ARIMA model
try:
    model = ARIMA(disasters_per_year, order=(1, 1, 0))
    model_fit = model.fit()
except Exception as e:
    print(f"Model fitting error: {e}")
    # Other 
    model = ARIMA(disasters_per_year, order=(0, 1, 1))
    model_fit = model.fit()
future_steps = 2030 - disasters_per_year.index[-1].year + 1  # From 2024 -2023
forecast_dates = pd.date_range(start=disasters_per_year.index[-1] + pd.DateOffset(days=1), periods=future_steps, freq='Y')
forecast = model_fit.forecast(steps=future_steps)
# Plot
plt.figure(figsize=(12, 6))
plt.plot(disasters_per_year.index,disasters_per_year, label='Actual', marker='o', color='blue')  
plt.plot(forecast_dates, forecast, label='Forecast', marker='x', color='red') 
plt.xlabel('Year') 
plt.ylabel('Number of disasters') 
plt.title('Forecast of disasters Through 2030')  
plt.legend() 
plt.grid(True) 
plt.show()

