import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

fema_data = pd.read_csv('us_disaster_declarations.csv')
fema_data['declaration_date'] = pd.to_datetime(fema_data['declaration_date'])
hurricane_data = fema_data[fema_data['incident_type'] == 'Hurricane']
hurricanes_per_year = hurricane_data.groupby(hurricane_data['declaration_date'].dt.year).size()

hurricanes_per_year.index = pd.to_datetime(hurricanes_per_year.index, format='%Y') + pd.offsets.YearEnd()

# Fit ARIMA 
try:
    model = ARIMA(hurricanes_per_year, order=(1, 1, 1))  
    model_fit = model.fit()
except Exception as e:
    print(f"Model fitting error: {e}")

future_steps = 2030 - hurricanes_per_year.index[-1].year + 1  # From 2024 -2023
forecast_dates = pd.date_range(start=hurricanes_per_year.index[-1] + pd.DateOffset(days=1), periods=future_steps, freq='Y')
forecast = model_fit.forecast(steps=future_steps)

#Plot
plt.figure(figsize=(12, 6))
plt.plot(hurricanes_per_year.index, hurricanes_per_year, label='Actual', marker='o', color='blue')  
plt.plot(forecast_dates, forecast, label='Forecast', marker='x', color='red') 
plt.xlabel('Year') 
plt.ylabel('Number of Hurricanes') 
plt.title('Forecast of Hurricanes Through 2030')  
plt.legend() 
plt.grid(True) 
plt.show()