# Ex.No: 6 HOLT WINTERS METHOD
### Date: 


## AIM:
To implement the Holt-Winters Method model on the Apple revenue dataset using Python and forecast future revenue values.
## DATASET
Apple 2009-2005
## SOFTWARE REQUIRED
Google Colab
## ALGORITHM
1. Import the necessary libraries such as pandas, matplotlib, numpy, sklearn, and statsmodels.
2. Load the Apple revenue CSV dataset into a DataFrame.
3. Set the year column as the index and perform initial data exploration.
4. Plot the Apple revenue time series data.
5. Scale the revenue data using MinMaxScaler.
6. Import the required statsmodels libraries for time series analysis.
7. Decompose the revenue data into additive components and plot them.
8. Split the dataset into training and testing data.
9. Create and train the Holt-Winters Exponential Smoothing model.
10. Predict the test dataset values using the trained model.
11. Calculate the Root Mean Squared Error (RMSE) to evaluate the model performance.
12. Calculate the mean and standard deviation of the scaled revenue dataset.
13. Fit the Holt-Winters model to the complete dataset and forecast future revenue values.
14. Plot the original revenue data and future predictions.


## PROGRAM
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Load dataset
data = pd.read_csv("Apple 2009-2024.csv")
data.head()

# Set year as index

data.set_index('year', inplace=True)
data.head()

# Convert Revenue column to numeric format

data['Revenue (millions)'] = (
    data['Revenue (millions)']
    .replace('[\$,]', '', regex=True)
    .astype(float)
)

# Plot revenue data
data['Revenue (millions)'].plot(figsize=(10,5))
plt.title('Apple Revenue Data')
plt.xlabel('Year')
plt.ylabel('Revenue (millions)')
plt.show()

# Scale the data and check for seasonality
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(
        data['Revenue (millions)'].values.reshape(-1, 1)
    ).flatten(),
    index=data.index
)
scaled_data.plot(figsize=(10,5))
plt.title('Scaled Revenue Data')
plt.xlabel('Year')
plt.ylabel('Scaled Revenue')
plt.show()

# Import seasonal decomposition and decompose data
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(
    data['Revenue (millions)'],
    model='additive',
    period=2
)
decomposition.plot()
plt.show()

# Split test and train data
scaled_data = scaled_data + 1
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

# Create Holt Linear Trend model
model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal=None,
    initialization_method='estimated'
).fit()

# Predict test data
test_predictions_add = model_add.forecast(
    len(test_data)
)
print(test_predictions_add)

# Visual evaluation
plt.figure(figsize=(10,5))
plt.plot(
    train_data.index,
    train_data,
    label='train_data'
)
plt.plot(
    test_data.index,
    test_data,
    label='test_data'
)
plt.plot(
    test_data.index,
    test_predictions_add,
    label='test_predictions_add'
)
plt.legend()
plt.title('Visual evaluation')
plt.xlabel('Year')
plt.ylabel('Scaled Revenue')
plt.show()

# RMSE value
rmse = np.sqrt(
    mean_squared_error(
        test_data,
        test_predictions_add
    )
)
print("RMSE:", rmse)

# Standard deviation and mean
print(
    "Standard Deviation:",
    np.sqrt(scaled_data.var())
)
print(
    "Mean:",
    scaled_data.mean()
)

# Final model and future prediction

final_model = ExponentialSmoothing(
    scaled_data,
    trend='add',
    seasonal=None,
    initialization_method='estimated'
).fit()
future_years = [2025, 2026, 2027, 2028]
final_predictions = final_model.forecast(4)

# Plot final prediction
plt.figure(figsize=(10,5))
plt.plot(
    scaled_data.index,
    scaled_data,
    label='scaled_data'
)
plt.plot(
    future_years,
    final_predictions,
    marker='o',
    label='final_predictions'
)
plt.legend()
plt.xlabel('Year')
plt.ylabel('Scaled Revenue')
plt.title('Prediction')
plt.show()
```

## OUTPUT

<img width="740" height="405" alt="image" src="https://github.com/user-attachments/assets/7cee7be2-1993-4a76-b538-aa9c5d2c0d7d" />
<img width="728" height="390" alt="image" src="https://github.com/user-attachments/assets/f8399a55-4a78-411d-8c97-1c41f17652a9" />


## RESULT:

Thus the program run successfully based on the Holt Winters Method model.
